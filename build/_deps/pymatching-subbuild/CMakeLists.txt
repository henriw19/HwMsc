# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.30.5-msvc23)

# Reject any attempt to use a toolchain file. We must not use one because
# we could be downloading it here. If the CMAKE_TOOLCHAIN_FILE environment
# variable is set, the cache variable will have been initialized from it.
unset(CMAKE_TOOLCHAIN_FILE CACHE)
unset(ENV{CMAKE_TOOLCHAIN_FILE})

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(pymatching-populate NONE)


# Pass through things we've already detected in the main project to avoid
# paying the cost of redetecting them again in ExternalProject_Add()
set(GIT_EXECUTABLE [==[D:/Anwendungen/Git/cmd/git.exe]==])
set(GIT_VERSION_STRING [==[2.40.0.windows.1]==])
set_property(GLOBAL PROPERTY _CMAKE_FindGit_GIT_EXECUTABLE_VERSION
  [==[D:/Anwendungen/Git/cmd/git.exe;2.40.0.windows.1]==]
)


include(ExternalProject)
ExternalProject_Add(pymatching-populate
                     "UPDATE_DISCONNECTED" "False" "GIT_REPOSITORY" "https://github.com/oscarhiggott/pymatching.git" "EXTERNALPROJECT_INTERNAL_ARGUMENT_SEPARATOR" "GIT_TAG" "40dcf8c01273ff7e23a7105d5cdc410ada067001"
                    SOURCE_DIR          "C:/Users/henri/Msc/build/_deps/pymatching-src"
                    BINARY_DIR          "C:/Users/henri/Msc/build/_deps/pymatching-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
                    USES_TERMINAL_PATCH     YES
)


