![](./dinosaur-logo.png)

# Dinosaur: differentiable dynamics for global atmospheric modeling

Authors: Jamie A. Smith, Dmitrii Kochkov, Peter Norgaard, Janni Yuval, Stephan Hoyer

Dinosaur is a spectral dynamical core for global atmospheric modeling written in
JAX:

- *Dynamics*: Dinosaur solves the shallow water equations, and the primitive equations (moist and dry) on sigma coordinates.
- *Auto-diff*: Dinosaur supports both forward- and backward-mode automatic differentiation in JAX.
- *Acceleration*: Dinosaur is designed to run efficiently on modern accelerator
hardware (GPU/TPU), including parallelization across multiple devices.

For more details, see our paper on [Neural General Circulation Models](https://arxiv.org/abs/2311.07222).

## Usage instructions

Dinosaur is an experimental research project that we are still working on
documenting.

We currently have examples replicating two standard test-cases for dynamical cores:

- [Baroclinic instability](https://nbviewer.org/github/neuralgcm/dinosaur/blob/main/notebooks/baroclinic_instability.ipynb)
- [Held-Suarez forcing](https://nbviewer.org/github/neuralgcm/dinosaur/blob/main/notebooks/held_suarez.ipynb)

We recommend running them using [Google Colab](https://colab.research.google.com/) with a GPU runtime.
You can also install Dinosaur locally: `pip install git+https://github.com/neuralgcm/dinosaur`

## See also

If you like Dinosaur, you might also like
[SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl), which
solves similar equations in Julia.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details. We are open to user
contributions, but please reach out (either on GitHub or by email) to coordinate
before starting significant work.

## License

Apache 2.0; see [`LICENSE`](LICENSE) for details.
