project(crash_repro_22754)

set(SOURCES main.cpp)

add_executable(crash_repro_22754 ${SOURCES})
target_link_libraries(crash_repro_22754 PRIVATE ttml)
