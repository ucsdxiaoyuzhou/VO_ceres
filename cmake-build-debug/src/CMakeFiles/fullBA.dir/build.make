# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

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
CMAKE_COMMAND = /home/xiaoyu/Downloads/clion-2017.1.2/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/xiaoyu/Downloads/clion-2017.1.2/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xiaoyu/Desktop/fullBA

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xiaoyu/Desktop/fullBA/cmake-build-debug

# Include any dependencies generated for this target.
include src/CMakeFiles/fullBA.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/fullBA.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/fullBA.dir/flags.make

src/CMakeFiles/fullBA.dir/draw.cpp.o: src/CMakeFiles/fullBA.dir/flags.make
src/CMakeFiles/fullBA.dir/draw.cpp.o: ../src/draw.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xiaoyu/Desktop/fullBA/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/fullBA.dir/draw.cpp.o"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fullBA.dir/draw.cpp.o -c /home/xiaoyu/Desktop/fullBA/src/draw.cpp

src/CMakeFiles/fullBA.dir/draw.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fullBA.dir/draw.cpp.i"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xiaoyu/Desktop/fullBA/src/draw.cpp > CMakeFiles/fullBA.dir/draw.cpp.i

src/CMakeFiles/fullBA.dir/draw.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fullBA.dir/draw.cpp.s"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xiaoyu/Desktop/fullBA/src/draw.cpp -o CMakeFiles/fullBA.dir/draw.cpp.s

src/CMakeFiles/fullBA.dir/draw.cpp.o.requires:

.PHONY : src/CMakeFiles/fullBA.dir/draw.cpp.o.requires

src/CMakeFiles/fullBA.dir/draw.cpp.o.provides: src/CMakeFiles/fullBA.dir/draw.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/fullBA.dir/build.make src/CMakeFiles/fullBA.dir/draw.cpp.o.provides.build
.PHONY : src/CMakeFiles/fullBA.dir/draw.cpp.o.provides

src/CMakeFiles/fullBA.dir/draw.cpp.o.provides.build: src/CMakeFiles/fullBA.dir/draw.cpp.o


src/CMakeFiles/fullBA.dir/Frame.cpp.o: src/CMakeFiles/fullBA.dir/flags.make
src/CMakeFiles/fullBA.dir/Frame.cpp.o: ../src/Frame.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xiaoyu/Desktop/fullBA/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/fullBA.dir/Frame.cpp.o"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fullBA.dir/Frame.cpp.o -c /home/xiaoyu/Desktop/fullBA/src/Frame.cpp

src/CMakeFiles/fullBA.dir/Frame.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fullBA.dir/Frame.cpp.i"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xiaoyu/Desktop/fullBA/src/Frame.cpp > CMakeFiles/fullBA.dir/Frame.cpp.i

src/CMakeFiles/fullBA.dir/Frame.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fullBA.dir/Frame.cpp.s"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xiaoyu/Desktop/fullBA/src/Frame.cpp -o CMakeFiles/fullBA.dir/Frame.cpp.s

src/CMakeFiles/fullBA.dir/Frame.cpp.o.requires:

.PHONY : src/CMakeFiles/fullBA.dir/Frame.cpp.o.requires

src/CMakeFiles/fullBA.dir/Frame.cpp.o.provides: src/CMakeFiles/fullBA.dir/Frame.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/fullBA.dir/build.make src/CMakeFiles/fullBA.dir/Frame.cpp.o.provides.build
.PHONY : src/CMakeFiles/fullBA.dir/Frame.cpp.o.provides

src/CMakeFiles/fullBA.dir/Frame.cpp.o.provides.build: src/CMakeFiles/fullBA.dir/Frame.cpp.o


src/CMakeFiles/fullBA.dir/Map.cpp.o: src/CMakeFiles/fullBA.dir/flags.make
src/CMakeFiles/fullBA.dir/Map.cpp.o: ../src/Map.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xiaoyu/Desktop/fullBA/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/fullBA.dir/Map.cpp.o"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fullBA.dir/Map.cpp.o -c /home/xiaoyu/Desktop/fullBA/src/Map.cpp

src/CMakeFiles/fullBA.dir/Map.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fullBA.dir/Map.cpp.i"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xiaoyu/Desktop/fullBA/src/Map.cpp > CMakeFiles/fullBA.dir/Map.cpp.i

src/CMakeFiles/fullBA.dir/Map.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fullBA.dir/Map.cpp.s"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xiaoyu/Desktop/fullBA/src/Map.cpp -o CMakeFiles/fullBA.dir/Map.cpp.s

src/CMakeFiles/fullBA.dir/Map.cpp.o.requires:

.PHONY : src/CMakeFiles/fullBA.dir/Map.cpp.o.requires

src/CMakeFiles/fullBA.dir/Map.cpp.o.provides: src/CMakeFiles/fullBA.dir/Map.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/fullBA.dir/build.make src/CMakeFiles/fullBA.dir/Map.cpp.o.provides.build
.PHONY : src/CMakeFiles/fullBA.dir/Map.cpp.o.provides

src/CMakeFiles/fullBA.dir/Map.cpp.o.provides.build: src/CMakeFiles/fullBA.dir/Map.cpp.o


src/CMakeFiles/fullBA.dir/MapPoint.cpp.o: src/CMakeFiles/fullBA.dir/flags.make
src/CMakeFiles/fullBA.dir/MapPoint.cpp.o: ../src/MapPoint.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xiaoyu/Desktop/fullBA/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/fullBA.dir/MapPoint.cpp.o"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fullBA.dir/MapPoint.cpp.o -c /home/xiaoyu/Desktop/fullBA/src/MapPoint.cpp

src/CMakeFiles/fullBA.dir/MapPoint.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fullBA.dir/MapPoint.cpp.i"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xiaoyu/Desktop/fullBA/src/MapPoint.cpp > CMakeFiles/fullBA.dir/MapPoint.cpp.i

src/CMakeFiles/fullBA.dir/MapPoint.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fullBA.dir/MapPoint.cpp.s"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xiaoyu/Desktop/fullBA/src/MapPoint.cpp -o CMakeFiles/fullBA.dir/MapPoint.cpp.s

src/CMakeFiles/fullBA.dir/MapPoint.cpp.o.requires:

.PHONY : src/CMakeFiles/fullBA.dir/MapPoint.cpp.o.requires

src/CMakeFiles/fullBA.dir/MapPoint.cpp.o.provides: src/CMakeFiles/fullBA.dir/MapPoint.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/fullBA.dir/build.make src/CMakeFiles/fullBA.dir/MapPoint.cpp.o.provides.build
.PHONY : src/CMakeFiles/fullBA.dir/MapPoint.cpp.o.provides

src/CMakeFiles/fullBA.dir/MapPoint.cpp.o.provides.build: src/CMakeFiles/fullBA.dir/MapPoint.cpp.o


src/CMakeFiles/fullBA.dir/utils.cpp.o: src/CMakeFiles/fullBA.dir/flags.make
src/CMakeFiles/fullBA.dir/utils.cpp.o: ../src/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xiaoyu/Desktop/fullBA/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/CMakeFiles/fullBA.dir/utils.cpp.o"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fullBA.dir/utils.cpp.o -c /home/xiaoyu/Desktop/fullBA/src/utils.cpp

src/CMakeFiles/fullBA.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fullBA.dir/utils.cpp.i"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xiaoyu/Desktop/fullBA/src/utils.cpp > CMakeFiles/fullBA.dir/utils.cpp.i

src/CMakeFiles/fullBA.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fullBA.dir/utils.cpp.s"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xiaoyu/Desktop/fullBA/src/utils.cpp -o CMakeFiles/fullBA.dir/utils.cpp.s

src/CMakeFiles/fullBA.dir/utils.cpp.o.requires:

.PHONY : src/CMakeFiles/fullBA.dir/utils.cpp.o.requires

src/CMakeFiles/fullBA.dir/utils.cpp.o.provides: src/CMakeFiles/fullBA.dir/utils.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/fullBA.dir/build.make src/CMakeFiles/fullBA.dir/utils.cpp.o.provides.build
.PHONY : src/CMakeFiles/fullBA.dir/utils.cpp.o.provides

src/CMakeFiles/fullBA.dir/utils.cpp.o.provides.build: src/CMakeFiles/fullBA.dir/utils.cpp.o


src/CMakeFiles/fullBA.dir/system.cpp.o: src/CMakeFiles/fullBA.dir/flags.make
src/CMakeFiles/fullBA.dir/system.cpp.o: ../src/system.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xiaoyu/Desktop/fullBA/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/CMakeFiles/fullBA.dir/system.cpp.o"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fullBA.dir/system.cpp.o -c /home/xiaoyu/Desktop/fullBA/src/system.cpp

src/CMakeFiles/fullBA.dir/system.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fullBA.dir/system.cpp.i"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xiaoyu/Desktop/fullBA/src/system.cpp > CMakeFiles/fullBA.dir/system.cpp.i

src/CMakeFiles/fullBA.dir/system.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fullBA.dir/system.cpp.s"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xiaoyu/Desktop/fullBA/src/system.cpp -o CMakeFiles/fullBA.dir/system.cpp.s

src/CMakeFiles/fullBA.dir/system.cpp.o.requires:

.PHONY : src/CMakeFiles/fullBA.dir/system.cpp.o.requires

src/CMakeFiles/fullBA.dir/system.cpp.o.provides: src/CMakeFiles/fullBA.dir/system.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/fullBA.dir/build.make src/CMakeFiles/fullBA.dir/system.cpp.o.provides.build
.PHONY : src/CMakeFiles/fullBA.dir/system.cpp.o.provides

src/CMakeFiles/fullBA.dir/system.cpp.o.provides.build: src/CMakeFiles/fullBA.dir/system.cpp.o


src/CMakeFiles/fullBA.dir/Mapviewer.cpp.o: src/CMakeFiles/fullBA.dir/flags.make
src/CMakeFiles/fullBA.dir/Mapviewer.cpp.o: ../src/Mapviewer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xiaoyu/Desktop/fullBA/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object src/CMakeFiles/fullBA.dir/Mapviewer.cpp.o"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fullBA.dir/Mapviewer.cpp.o -c /home/xiaoyu/Desktop/fullBA/src/Mapviewer.cpp

src/CMakeFiles/fullBA.dir/Mapviewer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fullBA.dir/Mapviewer.cpp.i"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xiaoyu/Desktop/fullBA/src/Mapviewer.cpp > CMakeFiles/fullBA.dir/Mapviewer.cpp.i

src/CMakeFiles/fullBA.dir/Mapviewer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fullBA.dir/Mapviewer.cpp.s"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xiaoyu/Desktop/fullBA/src/Mapviewer.cpp -o CMakeFiles/fullBA.dir/Mapviewer.cpp.s

src/CMakeFiles/fullBA.dir/Mapviewer.cpp.o.requires:

.PHONY : src/CMakeFiles/fullBA.dir/Mapviewer.cpp.o.requires

src/CMakeFiles/fullBA.dir/Mapviewer.cpp.o.provides: src/CMakeFiles/fullBA.dir/Mapviewer.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/fullBA.dir/build.make src/CMakeFiles/fullBA.dir/Mapviewer.cpp.o.provides.build
.PHONY : src/CMakeFiles/fullBA.dir/Mapviewer.cpp.o.provides

src/CMakeFiles/fullBA.dir/Mapviewer.cpp.o.provides.build: src/CMakeFiles/fullBA.dir/Mapviewer.cpp.o


src/CMakeFiles/fullBA.dir/optimizer.cpp.o: src/CMakeFiles/fullBA.dir/flags.make
src/CMakeFiles/fullBA.dir/optimizer.cpp.o: ../src/optimizer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xiaoyu/Desktop/fullBA/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object src/CMakeFiles/fullBA.dir/optimizer.cpp.o"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fullBA.dir/optimizer.cpp.o -c /home/xiaoyu/Desktop/fullBA/src/optimizer.cpp

src/CMakeFiles/fullBA.dir/optimizer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fullBA.dir/optimizer.cpp.i"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xiaoyu/Desktop/fullBA/src/optimizer.cpp > CMakeFiles/fullBA.dir/optimizer.cpp.i

src/CMakeFiles/fullBA.dir/optimizer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fullBA.dir/optimizer.cpp.s"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xiaoyu/Desktop/fullBA/src/optimizer.cpp -o CMakeFiles/fullBA.dir/optimizer.cpp.s

src/CMakeFiles/fullBA.dir/optimizer.cpp.o.requires:

.PHONY : src/CMakeFiles/fullBA.dir/optimizer.cpp.o.requires

src/CMakeFiles/fullBA.dir/optimizer.cpp.o.provides: src/CMakeFiles/fullBA.dir/optimizer.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/fullBA.dir/build.make src/CMakeFiles/fullBA.dir/optimizer.cpp.o.provides.build
.PHONY : src/CMakeFiles/fullBA.dir/optimizer.cpp.o.provides

src/CMakeFiles/fullBA.dir/optimizer.cpp.o.provides.build: src/CMakeFiles/fullBA.dir/optimizer.cpp.o


# Object files for target fullBA
fullBA_OBJECTS = \
"CMakeFiles/fullBA.dir/draw.cpp.o" \
"CMakeFiles/fullBA.dir/Frame.cpp.o" \
"CMakeFiles/fullBA.dir/Map.cpp.o" \
"CMakeFiles/fullBA.dir/MapPoint.cpp.o" \
"CMakeFiles/fullBA.dir/utils.cpp.o" \
"CMakeFiles/fullBA.dir/system.cpp.o" \
"CMakeFiles/fullBA.dir/Mapviewer.cpp.o" \
"CMakeFiles/fullBA.dir/optimizer.cpp.o"

# External object files for target fullBA
fullBA_EXTERNAL_OBJECTS =

../lib/libfullBA.so: src/CMakeFiles/fullBA.dir/draw.cpp.o
../lib/libfullBA.so: src/CMakeFiles/fullBA.dir/Frame.cpp.o
../lib/libfullBA.so: src/CMakeFiles/fullBA.dir/Map.cpp.o
../lib/libfullBA.so: src/CMakeFiles/fullBA.dir/MapPoint.cpp.o
../lib/libfullBA.so: src/CMakeFiles/fullBA.dir/utils.cpp.o
../lib/libfullBA.so: src/CMakeFiles/fullBA.dir/system.cpp.o
../lib/libfullBA.so: src/CMakeFiles/fullBA.dir/Mapviewer.cpp.o
../lib/libfullBA.so: src/CMakeFiles/fullBA.dir/optimizer.cpp.o
../lib/libfullBA.so: src/CMakeFiles/fullBA.dir/build.make
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libboost_regex.so
../lib/libfullBA.so: /usr/lib/libpcl_common.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../lib/libfullBA.so: /usr/lib/libpcl_kdtree.so
../lib/libfullBA.so: /usr/lib/libpcl_octree.so
../lib/libfullBA.so: /usr/lib/libpcl_search.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libqhull.so
../lib/libfullBA.so: /usr/lib/libpcl_surface.so
../lib/libfullBA.so: /usr/lib/libpcl_sample_consensus.so
../lib/libfullBA.so: /usr/lib/libOpenNI.so
../lib/libfullBA.so: /usr/lib/libOpenNI2.so
../lib/libfullBA.so: /usr/lib/libpcl_io.so
../lib/libfullBA.so: /usr/lib/libpcl_filters.so
../lib/libfullBA.so: /usr/lib/libpcl_features.so
../lib/libfullBA.so: /usr/lib/libpcl_keypoints.so
../lib/libfullBA.so: /usr/lib/libpcl_registration.so
../lib/libfullBA.so: /usr/lib/libpcl_segmentation.so
../lib/libfullBA.so: /usr/lib/libpcl_recognition.so
../lib/libfullBA.so: /usr/lib/libpcl_visualization.so
../lib/libfullBA.so: /usr/lib/libpcl_people.so
../lib/libfullBA.so: /usr/lib/libpcl_outofcore.so
../lib/libfullBA.so: /usr/lib/libpcl_tracking.so
../lib/libfullBA.so: /usr/lib/libpcl_apps.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libboost_regex.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libqhull.so
../lib/libfullBA.so: /usr/lib/libOpenNI.so
../lib/libfullBA.so: /usr/lib/libOpenNI2.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../lib/libfullBA.so: /usr/lib/libvtkGenericFiltering.so.5.8.0
../lib/libfullBA.so: /usr/lib/libvtkGeovis.so.5.8.0
../lib/libfullBA.so: /usr/lib/libvtkCharts.so.5.8.0
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.8
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
../lib/libfullBA.so: /usr/local/lib/libceres.a
../lib/libfullBA.so: /usr/local/lib/libgflags.a
../lib/libfullBA.so: /usr/lib/libpcl_common.so
../lib/libfullBA.so: /usr/lib/libpcl_kdtree.so
../lib/libfullBA.so: /usr/lib/libpcl_octree.so
../lib/libfullBA.so: /usr/lib/libpcl_search.so
../lib/libfullBA.so: /usr/lib/libpcl_surface.so
../lib/libfullBA.so: /usr/lib/libpcl_sample_consensus.so
../lib/libfullBA.so: /usr/lib/libpcl_io.so
../lib/libfullBA.so: /usr/lib/libpcl_filters.so
../lib/libfullBA.so: /usr/lib/libpcl_features.so
../lib/libfullBA.so: /usr/lib/libpcl_keypoints.so
../lib/libfullBA.so: /usr/lib/libpcl_registration.so
../lib/libfullBA.so: /usr/lib/libpcl_segmentation.so
../lib/libfullBA.so: /usr/lib/libpcl_recognition.so
../lib/libfullBA.so: /usr/lib/libpcl_visualization.so
../lib/libfullBA.so: /usr/lib/libpcl_people.so
../lib/libfullBA.so: /usr/lib/libpcl_outofcore.so
../lib/libfullBA.so: /usr/lib/libpcl_tracking.so
../lib/libfullBA.so: /usr/lib/libpcl_apps.so
../lib/libfullBA.so: /usr/lib/libvtkViews.so.5.8.0
../lib/libfullBA.so: /usr/lib/libvtkInfovis.so.5.8.0
../lib/libfullBA.so: /usr/lib/libvtkWidgets.so.5.8.0
../lib/libfullBA.so: /usr/lib/libvtkVolumeRendering.so.5.8.0
../lib/libfullBA.so: /usr/lib/libvtkHybrid.so.5.8.0
../lib/libfullBA.so: /usr/lib/libvtkParallel.so.5.8.0
../lib/libfullBA.so: /usr/lib/libvtkRendering.so.5.8.0
../lib/libfullBA.so: /usr/lib/libvtkImaging.so.5.8.0
../lib/libfullBA.so: /usr/lib/libvtkGraphics.so.5.8.0
../lib/libfullBA.so: /usr/lib/libvtkIO.so.5.8.0
../lib/libfullBA.so: /usr/lib/libvtkFiltering.so.5.8.0
../lib/libfullBA.so: /usr/lib/libvtkCommon.so.5.8.0
../lib/libfullBA.so: /usr/lib/libvtksys.so.5.8.0
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libopencv_nonfree.so.2.4.8
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libglog.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libgflags.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libspqr.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libcholmod.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libccolamd.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libcamd.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libcolamd.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libamd.so
../lib/libfullBA.so: /usr/lib/liblapack.so
../lib/libfullBA.so: /usr/lib/libf77blas.so
../lib/libfullBA.so: /usr/lib/libatlas.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.a
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/librt.so
../lib/libfullBA.so: /usr/lib/liblapack.so
../lib/libfullBA.so: /usr/lib/libf77blas.so
../lib/libfullBA.so: /usr/lib/libatlas.so
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.a
../lib/libfullBA.so: /usr/lib/x86_64-linux-gnu/librt.so
../lib/libfullBA.so: src/CMakeFiles/fullBA.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xiaoyu/Desktop/fullBA/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX shared library ../../lib/libfullBA.so"
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fullBA.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/fullBA.dir/build: ../lib/libfullBA.so

.PHONY : src/CMakeFiles/fullBA.dir/build

src/CMakeFiles/fullBA.dir/requires: src/CMakeFiles/fullBA.dir/draw.cpp.o.requires
src/CMakeFiles/fullBA.dir/requires: src/CMakeFiles/fullBA.dir/Frame.cpp.o.requires
src/CMakeFiles/fullBA.dir/requires: src/CMakeFiles/fullBA.dir/Map.cpp.o.requires
src/CMakeFiles/fullBA.dir/requires: src/CMakeFiles/fullBA.dir/MapPoint.cpp.o.requires
src/CMakeFiles/fullBA.dir/requires: src/CMakeFiles/fullBA.dir/utils.cpp.o.requires
src/CMakeFiles/fullBA.dir/requires: src/CMakeFiles/fullBA.dir/system.cpp.o.requires
src/CMakeFiles/fullBA.dir/requires: src/CMakeFiles/fullBA.dir/Mapviewer.cpp.o.requires
src/CMakeFiles/fullBA.dir/requires: src/CMakeFiles/fullBA.dir/optimizer.cpp.o.requires

.PHONY : src/CMakeFiles/fullBA.dir/requires

src/CMakeFiles/fullBA.dir/clean:
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src && $(CMAKE_COMMAND) -P CMakeFiles/fullBA.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/fullBA.dir/clean

src/CMakeFiles/fullBA.dir/depend:
	cd /home/xiaoyu/Desktop/fullBA/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xiaoyu/Desktop/fullBA /home/xiaoyu/Desktop/fullBA/src /home/xiaoyu/Desktop/fullBA/cmake-build-debug /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src /home/xiaoyu/Desktop/fullBA/cmake-build-debug/src/CMakeFiles/fullBA.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/fullBA.dir/depend
