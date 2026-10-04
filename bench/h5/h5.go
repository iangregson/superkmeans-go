package h5

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
)

var ErrUnsupported = errors.New("h5: unsupported HDF5 file")

type Kind string

const (
	F32 Kind = "f32"
	F64 Kind = "f64"
	I32 Kind = "i32"
	I64 Kind = "i64"
)

type Dataset struct {
	Name string
	Dims []int
	Kind Kind
	Elem int
	Addr int64
	Size int64
}

func (d Dataset) Rows() int { return d.Dims[0] }

func (d Dataset) RowLen() int {
	n := 1
	for _, v := range d.Dims[1:] {
		n *= v
	}
	return n
}

type File struct {
	path string
	f    *os.File
	size int64
	sets []Dataset
}

var signature = [8]byte{0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n'}

func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	h := &File{path: path, f: f, size: st.Size()}
	if err := h.walk(); err != nil {
		f.Close()
		return nil, err
	}
	return h, nil
}

func (h *File) Close() error { return h.f.Close() }

func (h *File) Path() string { return h.path }

func (h *File) Datasets() []Dataset {
	out := make([]Dataset, len(h.sets))
	copy(out, h.sets)
	return out
}

func (h *File) Lookup(name string) (Dataset, error) {
	for _, d := range h.sets {
		if d.Name == name {
			return d, nil
		}
	}
	return Dataset{}, fmt.Errorf("h5: %s: no dataset %q", h.path, name)
}

func (h *File) fail(format string, a ...any) error {
	return fmt.Errorf("%w: %s: %s", ErrUnsupported, h.path, fmt.Sprintf(format, a...))
}

func (h *File) at(off int64, n int) ([]byte, error) {
	if off < 0 || n < 0 || off+int64(n) > h.size {
		return nil, h.fail("read %d bytes at %d past %d-byte file", n, off, h.size)
	}
	b := make([]byte, n)
	if _, err := h.f.ReadAt(b, off); err != nil {
		return nil, err
	}
	return b, nil
}

type message struct {
	typ  uint16
	data []byte
}

func (h *File) header(addr int64) ([]message, error) {
	b, err := h.at(addr, 16)
	if err != nil {
		return nil, err
	}
	if b[0] != 1 {
		return nil, h.fail("object header version %d at %d, want 1", b[0], addr)
	}
	type chunk struct {
		off int64
		n   int64
	}
	chunks := []chunk{{addr + 16, int64(binary.LittleEndian.Uint32(b[8:]))}}
	var out []message
	for i := 0; i < len(chunks); i++ {
		data, err := h.at(chunks[i].off, int(chunks[i].n))
		if err != nil {
			return nil, err
		}
		for p := 0; p+8 <= len(data); {
			typ := binary.LittleEndian.Uint16(data[p:])
			size := int(binary.LittleEndian.Uint16(data[p+2:]))
			p += 8
			if p+size > len(data) {
				return nil, h.fail("message overruns header chunk")
			}
			m := message{typ: typ, data: data[p : p+size]}
			p += size
			if typ == 0x0010 {
				if len(m.data) < 16 {
					return nil, h.fail("short continuation message")
				}
				chunks = append(chunks, chunk{
					off: int64(binary.LittleEndian.Uint64(m.data[0:])),
					n:   int64(binary.LittleEndian.Uint64(m.data[8:])),
				})
				continue
			}
			out = append(out, m)
			if len(out) > 4096 {
				return nil, h.fail("too many header messages")
			}
		}
	}
	return out, nil
}

func (h *File) walk() error {
	sb, err := h.at(0, 96)
	if err != nil {
		return err
	}
	if [8]byte(sb[0:8]) != signature {
		return h.fail("no HDF5 signature")
	}
	if sb[8] != 0 {
		return h.fail("superblock version %d, want 0", sb[8])
	}
	if sb[13] != 8 || sb[14] != 8 {
		return h.fail("offset/length sizes %d/%d, want 8/8", sb[13], sb[14])
	}
	if binary.LittleEndian.Uint64(sb[24:]) != 0 {
		return h.fail("base address not zero")
	}
	if eof := int64(binary.LittleEndian.Uint64(sb[40:])); eof != h.size {
		return h.fail("superblock eof %d != file size %d", eof, h.size)
	}
	root := int64(binary.LittleEndian.Uint64(sb[64:]))
	msgs, err := h.header(root)
	if err != nil {
		return err
	}
	var btree, heap int64 = -1, -1
	for _, m := range msgs {
		if m.typ != 0x0011 {
			continue
		}
		if len(m.data) < 16 {
			return h.fail("short symbol table message")
		}
		btree = int64(binary.LittleEndian.Uint64(m.data[0:]))
		heap = int64(binary.LittleEndian.Uint64(m.data[8:]))
	}
	if btree < 0 || heap < 0 {
		return h.fail("root group has no symbol table")
	}
	heapData, err := h.heapData(heap)
	if err != nil {
		return err
	}
	if err := h.btree(btree, heapData, 0); err != nil {
		return err
	}
	sort.Slice(h.sets, func(i, j int) bool { return h.sets[i].Name < h.sets[j].Name })
	return nil
}

func (h *File) heapData(addr int64) (int64, error) {
	b, err := h.at(addr, 32)
	if err != nil {
		return 0, err
	}
	if string(b[0:4]) != "HEAP" {
		return 0, h.fail("no local heap at %d", addr)
	}
	if b[4] != 0 {
		return 0, h.fail("local heap version %d, want 0", b[4])
	}
	return int64(binary.LittleEndian.Uint64(b[24:])), nil
}

func (h *File) name(heapData, off int64) (string, error) {
	b, err := h.at(heapData+off, 256)
	if err != nil {
		return "", err
	}
	for i, c := range b {
		if c == 0 {
			return string(b[:i]), nil
		}
	}
	return "", h.fail("unterminated link name")
}

func (h *File) btree(addr, heapData int64, depth int) error {
	if depth > 8 {
		return h.fail("B-tree too deep")
	}
	b, err := h.at(addr, 24)
	if err != nil {
		return err
	}
	if string(b[0:4]) != "TREE" {
		return h.fail("no B-tree at %d", addr)
	}
	if b[4] != 0 {
		return h.fail("B-tree node type %d, want 0", b[4])
	}
	level := int(b[5])
	entries := int(binary.LittleEndian.Uint16(b[6:]))
	body, err := h.at(addr+24, entries*16+8)
	if err != nil {
		return err
	}
	for i := 0; i < entries; i++ {
		child := int64(binary.LittleEndian.Uint64(body[i*16+8:]))
		if level > 0 {
			if err := h.btree(child, heapData, depth+1); err != nil {
				return err
			}
			continue
		}
		if err := h.snod(child, heapData); err != nil {
			return err
		}
	}
	return nil
}

func (h *File) snod(addr, heapData int64) error {
	b, err := h.at(addr, 8)
	if err != nil {
		return err
	}
	if string(b[0:4]) != "SNOD" {
		return h.fail("no symbol node at %d", addr)
	}
	if b[4] != 1 {
		return h.fail("symbol node version %d, want 1", b[4])
	}
	n := int(binary.LittleEndian.Uint16(b[6:]))
	body, err := h.at(addr+8, n*40)
	if err != nil {
		return err
	}
	for i := 0; i < n; i++ {
		e := body[i*40:]
		nameOff := int64(binary.LittleEndian.Uint64(e[0:]))
		objAddr := int64(binary.LittleEndian.Uint64(e[8:]))
		if binary.LittleEndian.Uint32(e[16:]) != 0 {
			continue
		}
		name, err := h.name(heapData, nameOff)
		if err != nil {
			return err
		}
		ds, ok, err := h.dataset(name, objAddr)
		if err != nil {
			return err
		}
		if ok {
			h.sets = append(h.sets, ds)
		}
	}
	return nil
}

func (h *File) dataset(name string, addr int64) (Dataset, bool, error) {
	msgs, err := h.header(addr)
	if err != nil {
		return Dataset{}, false, err
	}
	ds := Dataset{Name: name, Addr: -1}
	var haveSpace, haveType, haveLayout bool
	for _, m := range msgs {
		switch m.typ {
		case 0x0001:
			dims, err := h.dataspace(name, m.data)
			if err != nil {
				return Dataset{}, false, err
			}
			ds.Dims, haveSpace = dims, true
		case 0x0003:
			kind, elem, err := h.datatype(name, m.data)
			if err != nil {
				return Dataset{}, false, err
			}
			ds.Kind, ds.Elem, haveType = kind, elem, true
		case 0x0008:
			a, size, err := h.layout(name, m.data)
			if err != nil {
				return Dataset{}, false, err
			}
			ds.Addr, ds.Size, haveLayout = a, size, true
		}
	}
	if !haveSpace || !haveType || !haveLayout {
		return Dataset{}, false, nil
	}
	want := int64(ds.Elem)
	for _, d := range ds.Dims {
		want *= int64(d)
	}
	if want != ds.Size {
		return Dataset{}, false, h.fail("dataset %q size mismatch", name)
	}
	if ds.Addr < 0 || ds.Addr+ds.Size > h.size {
		return Dataset{}, false, h.fail("dataset %q out of range", name)
	}
	return ds, true, nil
}

func (h *File) dataspace(name string, m []byte) ([]int, error) {
	if len(m) < 8 {
		return nil, h.fail("short dataspace for %q", name)
	}
	off := 8
	if m[0] == 2 {
		off = 4
	} else if m[0] != 1 {
		return nil, h.fail("dataspace version %d for %q", m[0], name)
	}
	n := int(m[1])
	if n < 1 || n > 4 {
		return nil, h.fail("dataset %q has %d dims", name, n)
	}
	if len(m) < off+8*n {
		return nil, h.fail("short dataspace dims for %q", name)
	}
	dims := make([]int, n)
	for i := range dims {
		v := binary.LittleEndian.Uint64(m[off+8*i:])
		if v > math.MaxInt32 {
			return nil, h.fail("dataset %q dim too large", name)
		}
		dims[i] = int(v)
	}
	return dims, nil
}

func (h *File) datatype(name string, m []byte) (Kind, int, error) {
	if len(m) < 8 {
		return "", 0, h.fail("short datatype for %q", name)
	}
	version, class := m[0]>>4, m[0]&0xf
	if version < 1 || version > 3 {
		return "", 0, h.fail("datatype version %d for %q", version, name)
	}
	size := int(binary.LittleEndian.Uint32(m[4:]))
	if m[1]&0x01 != 0 || m[1]&0x40 != 0 {
		return "", 0, h.fail("dataset %q is not little-endian fixed-size", name)
	}
	if len(m) < 12 || binary.LittleEndian.Uint16(m[8:]) != 0 || int(binary.LittleEndian.Uint16(m[10:])) != 8*size {
		return "", 0, h.fail("dataset %q uses a packed type", name)
	}
	switch class {
	case 0:
		if m[1]&0x08 == 0 {
			return "", 0, h.fail("dataset %q is unsigned", name)
		}
		switch size {
		case 4:
			return I32, 4, nil
		case 8:
			return I64, 8, nil
		}
	case 1:
		switch size {
		case 4:
			return F32, 4, nil
		case 8:
			return F64, 8, nil
		}
	}
	return "", 0, h.fail("dataset %q has class %d size %d", name, class, size)
}

func (h *File) layout(name string, m []byte) (int64, int64, error) {
	if len(m) < 2 {
		return 0, 0, h.fail("short layout for %q", name)
	}
	if m[0] != 3 && m[0] != 4 {
		return 0, 0, h.fail("layout version %d for %q", m[0], name)
	}
	if m[1] != 1 {
		return 0, 0, h.fail("dataset %q is not contiguous", name)
	}
	if len(m) < 18 {
		return 0, 0, h.fail("short contiguous layout for %q", name)
	}
	addr := binary.LittleEndian.Uint64(m[2:])
	size := binary.LittleEndian.Uint64(m[10:])
	if addr == math.MaxUint64 {
		return 0, 0, h.fail("dataset %q has no storage", name)
	}
	return int64(addr), int64(size), nil
}

func (h *File) rowBytes(ds Dataset, row, rows, want int) ([]byte, error) {
	if row < 0 || rows < 0 || row+rows > ds.Rows() {
		return nil, fmt.Errorf("h5: %s rows out of range", ds.Name)
	}
	if want != rows*ds.RowLen() {
		return nil, fmt.Errorf("h5: %s dst holds %d values, want %d", ds.Name, want, rows*ds.RowLen())
	}
	n := rows * ds.RowLen() * ds.Elem
	b := make([]byte, n)
	if n == 0 {
		return b, nil
	}
	if _, err := h.f.ReadAt(b, ds.Addr+int64(row)*int64(ds.RowLen()*ds.Elem)); err != nil {
		return nil, err
	}
	return b, nil
}

func (h *File) ReadF32(ds Dataset, row, rows int, dst []float32) error {
	if ds.Kind != F32 {
		return fmt.Errorf("h5: %s is %s, want f32", ds.Name, ds.Kind)
	}
	b, err := h.rowBytes(ds, row, rows, len(dst))
	if err != nil {
		return err
	}
	for i := range dst {
		dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return nil
}

func (h *File) ReadF64(ds Dataset, row, rows int, dst []float64) error {
	if ds.Kind != F64 {
		return fmt.Errorf("h5: %s is %s, want f64", ds.Name, ds.Kind)
	}
	b, err := h.rowBytes(ds, row, rows, len(dst))
	if err != nil {
		return err
	}
	for i := range dst {
		dst[i] = math.Float64frombits(binary.LittleEndian.Uint64(b[8*i:]))
	}
	return nil
}

func (h *File) ReadI32(ds Dataset, row, rows int, dst []int32) error {
	if ds.Kind != I32 {
		return fmt.Errorf("h5: %s is %s, want i32", ds.Name, ds.Kind)
	}
	b, err := h.rowBytes(ds, row, rows, len(dst))
	if err != nil {
		return err
	}
	for i := range dst {
		dst[i] = int32(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return nil
}

func (h *File) ReadI64(ds Dataset, row, rows int, dst []int64) error {
	if ds.Kind != I64 {
		return fmt.Errorf("h5: %s is %s, want i64", ds.Name, ds.Kind)
	}
	b, err := h.rowBytes(ds, row, rows, len(dst))
	if err != nil {
		return err
	}
	for i := range dst {
		dst[i] = int64(binary.LittleEndian.Uint64(b[8*i:]))
	}
	return nil
}
