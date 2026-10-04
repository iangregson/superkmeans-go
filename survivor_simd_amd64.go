//go:build goexperiment.simd && amd64

package superkmeans

import "simd/archsimd"

func survivorCount(dv []float32, thr float32, pos []uint32) int {
	count := 0
	thrv := archsimd.BroadcastFloat32x8(thr)
	i := 0
	n := len(dv)
	for ; i+8 <= n; i += 8 {
		v := archsimd.LoadFloat32x8(dv[i:])
		m := v.Less(thrv).ToInt32x8()
		if !m.IsZero() {
			var arr [8]uint32
			m.ToBits().StoreArray(&arr)
			for k := 0; k < 8; k++ {
				pos[count] = uint32(i + k)
				if arr[k] != 0 {
					count++
				}
			}
		}
	}
	for ; i < n; i++ {
		pos[count] = uint32(i)
		if dv[i] < thr {
			count++
		}
	}
	return count
}
