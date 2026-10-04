package superkmeans

import (
	"errors"
	"fmt"
)

var (
	ErrInvalidK         = errors.New("superkmeans: invalid k")
	ErrInvalidD         = errors.New("superkmeans: invalid d")
	ErrInvalidIters     = errors.New("superkmeans: invalid iters")
	ErrSamplingFraction = errors.New("superkmeans: sampling fraction must be in (0,1]")
	ErrMaxPoints        = errors.New("superkmeans: max points per cluster must be positive")
	ErrDataLength       = errors.New("superkmeans: data length mismatch")
	ErrTooFewPoints     = errors.New("superkmeans: n must be >= k")
	ErrTooFewSamples    = errors.New("superkmeans: not enough samples to train")
	ErrOutputLength     = errors.New("superkmeans: output length mismatch")
	ErrNonFinite        = errors.New("superkmeans: non-finite value in data")
	ErrInitialCentroids = errors.New("superkmeans: initial centroids length mismatch")
	ErrHierSampling     = errors.New("superkmeans: hierarchical requires sampling fraction 1")
	ErrHierClusterCount = errors.New("superkmeans: hierarchical produced wrong cluster count")
)

func dataLengthError(got, want int) error {
	return fmt.Errorf("%w: got %d, want %d", ErrDataLength, got, want)
}

func outputLengthError(got, want int) error {
	return fmt.Errorf("%w: got %d, want %d", ErrOutputLength, got, want)
}
