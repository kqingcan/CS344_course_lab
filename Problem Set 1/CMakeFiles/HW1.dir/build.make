# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1"

# Include any dependencies generated for this target.
include CMakeFiles/HW1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/HW1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/HW1.dir/flags.make

CMakeFiles/HW1.dir/HW1_generated_student_func.cu.o: CMakeFiles/HW1.dir/HW1_generated_student_func.cu.o.depend
CMakeFiles/HW1.dir/HW1_generated_student_func.cu.o: CMakeFiles/HW1.dir/HW1_generated_student_func.cu.o.cmake
CMakeFiles/HW1.dir/HW1_generated_student_func.cu.o: student_func.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir="/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/HW1.dir/HW1_generated_student_func.cu.o"
	cd "/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/CMakeFiles/HW1.dir" && /usr/bin/cmake -E make_directory "/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/CMakeFiles/HW1.dir//."
	cd "/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/CMakeFiles/HW1.dir" && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D "generated_file:STRING=/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/CMakeFiles/HW1.dir//./HW1_generated_student_func.cu.o" -D "generated_cubin_file:STRING=/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/CMakeFiles/HW1.dir//./HW1_generated_student_func.cu.o.cubin.txt" -P "/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/CMakeFiles/HW1.dir//HW1_generated_student_func.cu.o.cmake"

CMakeFiles/HW1.dir/main.o: CMakeFiles/HW1.dir/flags.make
CMakeFiles/HW1.dir/main.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/HW1.dir/main.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/HW1.dir/main.o -c "/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/main.cpp"

CMakeFiles/HW1.dir/main.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/HW1.dir/main.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/main.cpp" > CMakeFiles/HW1.dir/main.i

CMakeFiles/HW1.dir/main.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/HW1.dir/main.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/main.cpp" -o CMakeFiles/HW1.dir/main.s

CMakeFiles/HW1.dir/main.o.requires:

.PHONY : CMakeFiles/HW1.dir/main.o.requires

CMakeFiles/HW1.dir/main.o.provides: CMakeFiles/HW1.dir/main.o.requires
	$(MAKE) -f CMakeFiles/HW1.dir/build.make CMakeFiles/HW1.dir/main.o.provides.build
.PHONY : CMakeFiles/HW1.dir/main.o.provides

CMakeFiles/HW1.dir/main.o.provides.build: CMakeFiles/HW1.dir/main.o


CMakeFiles/HW1.dir/reference_calc.o: CMakeFiles/HW1.dir/flags.make
CMakeFiles/HW1.dir/reference_calc.o: reference_calc.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/HW1.dir/reference_calc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/HW1.dir/reference_calc.o -c "/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/reference_calc.cpp"

CMakeFiles/HW1.dir/reference_calc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/HW1.dir/reference_calc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/reference_calc.cpp" > CMakeFiles/HW1.dir/reference_calc.i

CMakeFiles/HW1.dir/reference_calc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/HW1.dir/reference_calc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/reference_calc.cpp" -o CMakeFiles/HW1.dir/reference_calc.s

CMakeFiles/HW1.dir/reference_calc.o.requires:

.PHONY : CMakeFiles/HW1.dir/reference_calc.o.requires

CMakeFiles/HW1.dir/reference_calc.o.provides: CMakeFiles/HW1.dir/reference_calc.o.requires
	$(MAKE) -f CMakeFiles/HW1.dir/build.make CMakeFiles/HW1.dir/reference_calc.o.provides.build
.PHONY : CMakeFiles/HW1.dir/reference_calc.o.provides

CMakeFiles/HW1.dir/reference_calc.o.provides.build: CMakeFiles/HW1.dir/reference_calc.o


CMakeFiles/HW1.dir/compare.o: CMakeFiles/HW1.dir/flags.make
CMakeFiles/HW1.dir/compare.o: compare.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/HW1.dir/compare.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/HW1.dir/compare.o -c "/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/compare.cpp"

CMakeFiles/HW1.dir/compare.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/HW1.dir/compare.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/compare.cpp" > CMakeFiles/HW1.dir/compare.i

CMakeFiles/HW1.dir/compare.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/HW1.dir/compare.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/compare.cpp" -o CMakeFiles/HW1.dir/compare.s

CMakeFiles/HW1.dir/compare.o.requires:

.PHONY : CMakeFiles/HW1.dir/compare.o.requires

CMakeFiles/HW1.dir/compare.o.provides: CMakeFiles/HW1.dir/compare.o.requires
	$(MAKE) -f CMakeFiles/HW1.dir/build.make CMakeFiles/HW1.dir/compare.o.provides.build
.PHONY : CMakeFiles/HW1.dir/compare.o.provides

CMakeFiles/HW1.dir/compare.o.provides.build: CMakeFiles/HW1.dir/compare.o


# Object files for target HW1
HW1_OBJECTS = \
"CMakeFiles/HW1.dir/main.o" \
"CMakeFiles/HW1.dir/reference_calc.o" \
"CMakeFiles/HW1.dir/compare.o"

# External object files for target HW1
HW1_EXTERNAL_OBJECTS = \
"/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/CMakeFiles/HW1.dir/HW1_generated_student_func.cu.o"

HW1: CMakeFiles/HW1.dir/main.o
HW1: CMakeFiles/HW1.dir/reference_calc.o
HW1: CMakeFiles/HW1.dir/compare.o
HW1: CMakeFiles/HW1.dir/HW1_generated_student_func.cu.o
HW1: CMakeFiles/HW1.dir/build.make
HW1: /usr/local/cuda-9.0/lib64/libcudart_static.a
HW1: /usr/lib/x86_64-linux-gnu/librt.so
HW1: CMakeFiles/HW1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable HW1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/HW1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/HW1.dir/build: HW1

.PHONY : CMakeFiles/HW1.dir/build

CMakeFiles/HW1.dir/requires: CMakeFiles/HW1.dir/main.o.requires
CMakeFiles/HW1.dir/requires: CMakeFiles/HW1.dir/reference_calc.o.requires
CMakeFiles/HW1.dir/requires: CMakeFiles/HW1.dir/compare.o.requires

.PHONY : CMakeFiles/HW1.dir/requires

CMakeFiles/HW1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/HW1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/HW1.dir/clean

CMakeFiles/HW1.dir/depend: CMakeFiles/HW1.dir/HW1_generated_student_func.cu.o
	cd "/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1" "/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1" "/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1" "/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1" "/home/volcano/Desktop/cs344/Problem Sets/Problem Set 1/CMakeFiles/HW1.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/HW1.dir/depend
