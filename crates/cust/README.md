# Cust 

Featureful, Safe, and Fast CUDA driver API library heavily derived from RustaCUDA.

Cust is a fork of rustacuda with a lot of API changes, added functions, etc. Big thanks to everyone who worked on RustaCUDA!

## Why not just contribute to RustaCUDA?

This library is currently maintained for CUDA support in the onda fluid engine, therefore i wanted to have a deeper level of control
over such an integral part of onda. This allows me to completely break and redesign the API from the ground up, change things to allow for
more features like CUDA graphs, rename stuff, etc. However, this library can be used outside of onda just fine.
