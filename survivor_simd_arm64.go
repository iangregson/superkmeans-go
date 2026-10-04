//go:build goexperiment.simd && arm64

package superkmeans

import "simd/archsimd"

func survivorCount(dv []float32, thr float32, pos []uint32) int {
	count := 0
	thrv := archsimd.BroadcastFloat32x4(thr)
	i := 0
	n := len(dv)
	for ; i+4 <= n; i += 4 {
		v := archsimd.LoadFloat32x4(dv[i:])
		m := v.Less(thrv).ToInt32x4()
		if m.ReduceMin() != 0 {
			var arr [4]uint32
			m.ToBits().StoreArray(&arr)
			pos[count] = uint32(i)
			if arr[0] != 0 {
				count++
			}
			pos[count] = uint32(i + 1)
			if arr[1] != 0 {
				count++
			}
			pos[count] = uint32(i + 2)
			if arr[2] != 0 {
				count++
			}
			pos[count] = uint32(i + 3)
			if arr[3] != 0 {
				count++
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
