# The log 

Should make a new directory every time a simulation runs with the logging active.

Struct that holds the info needed for logging the simulation, dumping the data, and making figures and such.

Compose the recorder into the simulation? YES

Goes with the general opinion that the science runner script is responsible for all components of the simulation (lattice, logger/recorder, model somehow?)

Decouple logging logic from the simulation logic. 
The Simulation needs a logger, it is going to be sending all of the information is works with to that logger, and then the logger decides what gets written where.
