# Dinosaur: Differentiable Dynamics for Global Atmospheric Modeling 🦖

Authors: Jamie A. Smith, Dmitrii Kochkov, Peter Norgaard, Janni Yuval, Stephan Hoyer

Dinosaur is an old-fashioned (some might say prehistoric) dynamical core for global atmospheric modeling, re-written in [JAX](https://github.com/jax-ml/jax) to meet the needs of modern AI weather/climate models:

- *Dynamics*: Dinosaur uses spectral methods to solve the shallow water equations and the primitive equations (moist and dry) on sigma coordinates.
- *Auto-diff*: Dinosaur supports both forward- and backward-mode automatic differentiation in JAX. This enables "online training" of hybrid AI/physics models.
- *Acceleration*: Dinosaur is designed to run efficiently on modern accelerator
hardware (GPU/TPU), including parallelization across multiple devices.

For more details, see our paper [Neural General Circulation Models for Weather and Climate](https://www.nature.com/articles/s41586-024-07744-y).

## Usage instructions

Dinosaur is an experimental research project that we are still working on
documenting.

We currently have three notebooks illustrating how to use Dinosaur:

- [Baroclinic instability](https://nbviewer.org/github/neuralgcm/dinosaur/blob/main/notebooks/baroclinic_instability.ipynb)
- [Held-Suarez forcing](https://nbviewer.org/github/neuralgcm/dinosaur/blob/main/notebooks/held_suarez.ipynb)
- [Weather forecast on ERA5](https://nbviewer.org/github/neuralgcm/dinosaur/blob/main/notebooks/weather_forecast_on_era5.ipynb)

We recommend running them using [Google Colab](https://colab.research.google.com/) with a GPU runtime.
You can also install Dinosaur locally: `pip install dinosaur`

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
