#include <fstream>
#include <vector>
#include <iterator>

// Structure to hold information about a single atom
struct Atom {
    int id;
    int type;
    double x, y, z;
    // Add other relevant properties as needed (e.g., vx, vy, vz, fx, fy, fz, etc.)
};

// Structure to hold data for a single snapshot (timestep)
struct Snapshot {
    long long timestep;
    int num_atoms;
    double box_x0[3];
    double L[3];
    std::vector<double> coords;
    std::vector<int> types;
    //std::vector<Atom> atoms;

    Snapshot() : timestep(0), num_atoms(0),
                 box_x0{0.0,0.0,0.0}, L{0.0,0.0,0.0} {
    }

    bool load(std::ifstream &);
};

class DumpfileIterator {
public:
    // Iterator traits
    using value_type = Snapshot;
    using difference_type = std::ptrdiff_t;
    using pointer = const Snapshot*;
    using reference = const Snapshot&;
    using iterator_category = std::input_iterator_tag;

    // Default constructor for the end iterator
    DumpfileIterator() : file_stream_(nullptr) {}

    // Constructor for the begin iterator
    explicit DumpfileIterator(std::ifstream& fs) : file_stream_(&fs) {
        if (file_stream_ && file_stream_->is_open()) {
            read_snap(); // Read the first snapshot upon construction
        }
    }

    // Dereference operator
    const Snapshot& operator*() const {
        return current_;
    }

    // Pre-increment operator
    DumpfileIterator& operator++() {
        read_snap();
        return *this;
    }

    // Post-increment operator
    DumpfileIterator operator++(int) {
        DumpfileIterator temp = *this;
        read_snap();
        return temp;
    }

    // Equality operator
    bool operator==(const DumpfileIterator& other) const {
        return file_stream_ == other.file_stream_; // && current_ == other.current_;
    }

    // Inequality operator
    bool operator!=(const DumpfileIterator& other) const {
        return !(*this == other);
    }

private:
    std::ifstream* file_stream_;
    Snapshot current_;

    // Helper function to read a line
    void read_snap() {
        if (file_stream_ && current_.load(*file_stream_)) {
            // Line successfully read
        } else {
            // End of file or error, set stream to nullptr to signify end
            file_stream_ = nullptr;
            current_ = Snapshot(); // Clear line for end iterator
        }
    }
};

class Dumpfile {
  public:
    //Dumpfile(std::ifstream& fs) : file(fs) {}
    Dumpfile(const std::string &filename) : file(filename) { }
    bool is_open() {
        return file.is_open();
    }
    DumpfileIterator begin() {
        return DumpfileIterator(file);
    }
    DumpfileIterator end() {
        return DumpfileIterator();
    }
    std::ifstream file;
};
