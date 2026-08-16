#include <iostream>
#include <vector>

double calculate_mean(const std::vector<int>& pixels) {
    long long sum = 0;

    for (int value : pixels) {
        sum += value;
    }

    return static_cast<double>(sum) / pixels.size();
}

void apply_threshold(std::vector<int>& pixels, int threshold_value) {
    for (int& value : pixels) {
        value = value >= threshold_value ? 255 : 0;
    }
}

int main() {
    std::cout << "Day44: C++ computer vision begins!\n";

    int pixel_value = 10;
    int& pixel_reference = pixel_value;
    int* pixel_pointer = &pixel_value;

    pixel_reference = 20;
    *pixel_pointer = 30;

    std::cout << "Value after reference and pointer changes: "
              << pixel_value << '\n';
    std::cout << "Address stored by pointer: " << pixel_pointer << '\n';
    std::cout << "Address of original value: " << &pixel_value << '\n';

    std::vector<int> pixels{10, 80, 127, 128, 180, 255};
    std::cout << "Mean before threshold: " << calculate_mean(pixels) << '\n';

    apply_threshold(pixels, 150);

    std::cout << "Threshold result: ";
    for (int value : pixels) {
        std::cout << value << ' ';
    }
    std::cout << '\n';

    return 0;
}
