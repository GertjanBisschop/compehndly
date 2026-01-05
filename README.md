# Aim

This library acts as a community-maintained registry containing standardised functions to transform raw Personal Exposure Health (PEH) data.

# Calling registered functions

Functions registered in the library can be called as `compehndly.example_function["0.0.1"]` where `example_function` is a function with potentially different versions. Leaving out the version will result in calling the latest version of the function.