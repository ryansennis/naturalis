# Create naturalis C++ library
add_library(naturalis_cpp STATIC
  dynamics/dynamics.cpp
  integrators/integrators.cpp
  propagator/propagator.cpp
)

# Set include directories
# Headers use "naturalis/..." includes, so we need to set up the include path
# such that "naturalis/orbit.hpp" resolves to the actual file locations
# We add src/C++ as base, then create directory structure or adjust includes
target_include_directories(naturalis_cpp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/orbit
  ${CMAKE_CURRENT_SOURCE_DIR}/dynamics
  ${CMAKE_CURRENT_SOURCE_DIR}/integrators
  ${CMAKE_CURRENT_SOURCE_DIR}/propagator
  ${Eigen3_SOURCE_DIR}
)

# Link Eigen
target_link_libraries(naturalis_cpp PUBLIC Eigen3::Eigen)

# Set target properties
set_target_properties(naturalis_cpp PROPERTIES
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN ON
)
