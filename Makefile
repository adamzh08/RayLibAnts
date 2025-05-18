# Compiler
CC = /opt/intel/oneapi/compiler/latest/bin/icx
#CC = gcc

# Compiler flags
CFLAGS = -lm -O3 -ffast-math -lraylib -ldl -lpthread

# Target executable
TARGET = world

# Source files
SRCS = world.c

# Compile source files into object files
all: $(SRCS)
	$(CC) -o $(TARGET) $(SRCS) $(CFLAGS)

# Clean up build files
clean:
	rm -f $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)