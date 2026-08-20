#include <iostream>
#include <memory>
#include <string>
#include <utility>

struct TrackedResource {
    explicit TrackedResource(std::string resource_name)
        : name(std::move(resource_name)) {
        std::cout << "construct: " << name << '\n';
    }

    ~TrackedResource() {
        std::cout << "destroy: " << name << '\n';
    }

    std::string name;
};

void unique_pointer_demo() {
    std::cout << "[unique_ptr]\n";
    auto owner = std::make_unique<TrackedResource>("camera-buffer");
    std::unique_ptr<TrackedResource> next_owner = std::move(owner);
    std::cout << "source empty: " << std::boolalpha << (owner == nullptr) << '\n';
    std::cout << "new owner: " << next_owner->name << '\n';
}

void shared_pointer_demo() {
    std::cout << "[shared_ptr]\n";
    auto first = std::make_shared<TrackedResource>("shared-config");
    std::cout << "count: " << first.use_count() << '\n';
    {
        std::shared_ptr<TrackedResource> second = first;
        std::cout << "count: " << first.use_count() << '\n';
        std::cout << "second sees: " << second->name << '\n';
    }
    std::cout << "count: " << first.use_count() << '\n';
}

int main() {
    unique_pointer_demo();
    shared_pointer_demo();
    std::cout << "DAY48_SMART_POINTERS_OK\n";
    return 0;
}
