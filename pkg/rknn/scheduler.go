package rknn

import (
	"math"
	"math/rand"
)

const (
	numTrainTimesteps = 1000
	betaStart         = 0.00085
	betaEnd           = 0.012
	sigmaData         = 0.5
	timestepScaling   = 10.0
)

type Scheduler struct {
	Mode          string // "lcm" or "euler"
	Steps         int
	AlphasCumprod []float64
	Sigmas        []float64
	Timesteps     []int
	stepIndex     int
	rng           *rand.Rand
}

func NewScheduler(steps int, mode string) *Scheduler {
	if steps < 1 {
		steps = 4
	}
	if mode == "" {
		mode = "lcm"
	}

	// 1. Compute scaled-linear betas and cumulative alphas over 1000 training steps
	alphasCumprod := make([]float64, numTrainTimesteps)
	sqrtStart := math.Sqrt(betaStart)
	sqrtEnd := math.Sqrt(betaEnd)
	prod := 1.0
	for i := 0; i < numTrainTimesteps; i++ {
		sqrtBeta := sqrtStart + (sqrtEnd-sqrtStart)*float64(i)/float64(numTrainTimesteps-1)
		beta := sqrtBeta * sqrtBeta
		prod *= (1.0 - beta)
		alphasCumprod[i] = prod
	}

	s := &Scheduler{
		Mode:          mode,
		Steps:         steps,
		AlphasCumprod: alphasCumprod,
		rng:           rand.New(rand.NewSource(42)),
	}

	if mode == "lcm" {
		s.initLCM()
	} else {
		s.initEuler()
	}
	return s
}

func (s *Scheduler) Reset(seed int64) {
	s.stepIndex = 0
	s.rng = rand.New(rand.NewSource(seed))
}

func (s *Scheduler) initLCM() {
	const originalInferenceSteps = 50
	stepRatio := numTrainTimesteps / originalInferenceSteps // 20

	// Build the 50-step schedule: [19, 39, ..., 999]
	lcmOriginTimesteps := make([]int, originalInferenceSteps)
	for i := 0; i < originalInferenceSteps; i++ {
		lcmOriginTimesteps[i] = (i+1)*stepRatio - 1
	}

	// Subsample s.Steps evenly from the 50-step schedule in reverse order
	steps := s.Steps
	if steps > originalInferenceSteps {
		steps = originalInferenceSteps
	}
	s.Timesteps = make([]int, steps)
	if steps == 1 {
		s.Timesteps[0] = lcmOriginTimesteps[originalInferenceSteps-1]
		return
	}
	for i := 0; i < steps; i++ {
		pos := float64(i) * float64(originalInferenceSteps-1) / float64(steps-1)
		idx := originalInferenceSteps - 1 - int(math.Round(pos))
		if idx < 0 {
			idx = 0
		}
		if idx >= originalInferenceSteps {
			idx = originalInferenceSteps - 1
		}
		s.Timesteps[i] = lcmOriginTimesteps[idx]
	}
}

func (s *Scheduler) initEuler() {
	s.Timesteps = make([]int, s.Steps)
	s.Sigmas = make([]float64, s.Steps+1)

	for i := 0; i < s.Steps; i++ {
		// Linearly spaced timesteps from 999 down to 0
		tFloat := float64(numTrainTimesteps-1) * float64(s.Steps-1-i) / float64(maxInt(1, s.Steps-1))
		t := int(math.Round(tFloat))
		s.Timesteps[i] = t

		alphaBar := s.AlphasCumprod[t]
		s.Sigmas[i] = math.Sqrt((1.0 - alphaBar) / alphaBar)
	}
	s.Sigmas[s.Steps] = 0.0
}

// InitNoiseSigma returns the initial noise multiplier for sampling x_T ~ N(0, sigma^2 I).
func (s *Scheduler) InitNoiseSigma() float64 {
	if s.Mode == "lcm" {
		return 1.0 // LCM starts with unit Gaussian noise at t = 999
	}
	if len(s.Sigmas) > 0 {
		return s.Sigmas[0]
	}
	return 1.0
}

// NextTimestep returns the current discrete timestep to feed into the UNet.
func (s *Scheduler) NextTimestep() (int, bool) {
	if s.stepIndex >= len(s.Timesteps) {
		return 0, false
	}
	return s.Timesteps[s.stepIndex], true
}

// ScaleModelInput scales the latent tensor before passing it to the UNet.
func (s *Scheduler) ScaleModelInput(sample []float32, timestep int) []float32 {
	out := make([]float32, len(sample))
	if s.Mode == "lcm" {
		// LCM uses unscaled latents directly
		copy(out, sample)
		return out
	}
	// Euler scales by 1 / sqrt(sigma^2 + 1)
	sigma := s.Sigmas[s.stepIndex]
	scale := float32(1.0 / math.Sqrt(sigma*sigma+1.0))
	for i, v := range sample {
		out[i] = v * scale
	}
	return out
}

// Step applies one denoising update and advances the internal step counter.
func (s *Scheduler) Step(modelOutput []float32, timestep int, sample []float32) []float32 {
	defer func() { s.stepIndex++ }()

	if s.Mode == "lcm" {
		return s.stepLCM(modelOutput, timestep, sample)
	}
	return s.stepEuler(modelOutput, sample)
}

func (s *Scheduler) stepLCM(eps []float32, timestep int, sample []float32) []float32 {
	alphaProdT := s.AlphasCumprod[timestep]
	betaProdT := 1.0 - alphaProdT
	sqrtAlphaT := math.Sqrt(alphaProdT)
	sqrtBetaT := math.Sqrt(betaProdT)

	// Determine previous timestep's cumulative alpha
	alphaProdTPrev := 1.0
	isFinalStep := s.stepIndex >= len(s.Timesteps)-1
	if !isFinalStep {
		prevT := s.Timesteps[s.stepIndex+1]
		alphaProdTPrev = s.AlphasCumprod[prevT]
	}
	betaProdTPrev := 1.0 - alphaProdTPrev

	// Consistency model boundary condition scalings (c_skip, c_out)
	scaledT := float64(timestep) * timestepScaling
	cSkip := (sigmaData * sigmaData) / (scaledT*scaledT + sigmaData*sigmaData)
	cOut := scaledT / math.Sqrt(scaledT*scaledT+sigmaData*sigmaData)

	nextSample := make([]float32, len(sample))
	sqrtAlphaPrev := math.Sqrt(alphaProdTPrev)
	sqrtBetaPrev := math.Sqrt(betaProdTPrev)

	for i := range sample {
		x := float64(sample[i])
		e := float64(eps[i])

		// 1. Predict clean x_0 from epsilon prediction
		predX0 := (x - sqrtBetaT*e) / sqrtAlphaT

		// 2. Apply consistency boundary condition
		denoised := cSkip*x + cOut*predX0

		// 3. Re-noise to previous timestep unless on the final step
		if isFinalStep {
			nextSample[i] = float32(denoised)
		} else {
			noise := s.rng.NormFloat64()
			nextSample[i] = float32(sqrtAlphaPrev*denoised + sqrtBetaPrev*noise)
		}
	}
	return nextSample
}

func (s *Scheduler) stepEuler(eps []float32, sample []float32) []float32 {
	sigma := s.Sigmas[s.stepIndex]
	sigmaNext := s.Sigmas[s.stepIndex+1]
	dt := sigmaNext - sigma

	nextSample := make([]float32, len(sample))
	for i := range sample {
		// For epsilon prediction: pred_x0 = x - sigma * eps -> derivative = (x - pred_x0) / sigma = eps
		d := float64(eps[i])
		nextSample[i] = float32(float64(sample[i]) + d*dt)
	}
	return nextSample
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
