// sample.cpp
extern "C" {
    // Export a function 'add' that takes two doubles and returns their sum.
    double add(double a, double b) {
        return a + b;
    }
}
