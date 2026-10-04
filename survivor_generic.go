//go:build !goexperiment.simd || (!arm64 && !amd64)

package superkmeans

func survivorCount(dv []float32, thr float32, pos []uint32) int {
	count := 0
	for i, v := range dv {
		pos[count] = uint32(i)
		if v < thr {
			count++
		}
	}
	return count
}
