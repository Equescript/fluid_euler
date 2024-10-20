cmake_minimum_required(VERSION 3.22)

project(fluid_euler)

add_library(${PROJECT_NAME} SHARED
    src/main.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC "./include")
