# helper script that finds GLUT, either from the system install (linux), or from the included, precompiled binaries (windows)
# Note we *intentionally* do not use the file name of "FindGLUT.cmake" because we want to call the system-provided FindGLUT later on, we just set up some paths, where required

# legacy gl vs glvnd/glx
if (POLICY CMP0072)
  cmake_policy(SET CMP0072 NEW)
endif()

if (WIN32)
   # The default cmake-FindGLUT.cmake script will automatically search in 
   # - ${GLUT_ROOT_PATH}/Release (fro the lib)
   # - ${GLUT_ROOT_PATH}/include 
   # ... ie, setting this search path _should_ make the default script find the
   # right stuff, and set the right variables
   set(GLUT_ROOT_PATH "${CMAKE_CURRENT_LIST_DIR}/../3rdParty/freeglut")
endif()


find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
