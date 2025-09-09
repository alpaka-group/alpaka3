# N-Body Simulation

This example simulates a number of particles by calculating pair-wise gravity between them.

To run this example, first build it by executing in the repository's root directory:

```bash
$ mkdir build && cd build
$ ccmake .. # optionally adjust cmake options to use an accelerator etc.
$ cmake --build . --target=nBody
$ ./example/nBody/nBody
```

There are some options for the example executable:

- `-n numParticles`: Number of particles. Default: 512
- `-t numTimeSteps`: Number of time steps that the simulation is run for. Default: 1000
- `-d dt`: Delta t for the timesteps. Default: 0.001
- `-p`: Write pngs of the particles to disk. Has no effect when pngwriter is not installed. Default: off
- `-h`: Print this help message

For example,

```bash
$ ./example/nBody/nBody -n 8192 -t 2000 -d 0.0001 -p
```

would execute the example with 8192 particles for 2000 timesteps and dt = 0.0001, outputting pngs if PNGWriter is
installed. The resulting images are saved in the executable's directory. You can use the following ffmpeg command to
stitch the images together into a video file:

```bash 
$ ffmpeg -framerate 60 -i particles_%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 -preset fast particles.mkv
```
