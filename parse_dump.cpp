/* Parse a LAMMPS dump file.
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
50000
ITEM: BOX BOUNDS pp pp pp
0.0000000000000000e+00 1.3273560525675299e+02
0.0000000000000000e+00 1.3273560525675299e+02
0.0000000000000000e+00 1.3273560525675299e+02
ITEM: ATOMS id type xs ys zs
4181 1 0.0643069 0.054575 0.00234367
32749 1 0.0503662 0.0328496 0.0120233
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

#include "snapshot.h"

// Function to parse a LAMMPS dump file
bool Snapshot::load(std::ifstream &file) {
    int state = 0;

    std::string line;
    //std::vector<std::string> atom_properties;
    *this = Snapshot();
    int natoms = 0;

    while (std::getline(file, line)) {
        if (line.rfind("ITEM: TIMESTEP", 0) == 0) {
            if(natoms > 0) {
                if(natoms != this->num_atoms) {
                    std::cerr << "Incorrect frame: "
                      << natoms << " of " << this->num_atoms
                      << " atoms." << std::endl;
                    return false;
                }
            }
            std::stringstream ss(line);
            std::string temp;
            ss >> temp >> temp >> this->timestep;
            state = 0; continue;
        }
        if (line.rfind("ITEM: NUMBER OF ATOMS", 0) == 0) { // check if line starts with "ITEM: NUMBER OF ATOMS"
            std::string temp_line;
            std::getline(file, temp_line); // Read x bounds
            std::stringstream ss(temp_line);
            ss >> this->num_atoms;
            this->types.resize(this->num_atoms);
            this->coords.resize(this->num_atoms*3);
            state = 0; continue;
        }
        if (line.rfind("ITEM: BOX BOUNDS", 0) == 0) { // check if line starts with "ITEM: BOX BOUNDS"
            // Assuming "pp pp pp" for periodic boundaries
            double xmin, xmax;
            double ymin, ymax;
            double zmin, zmax;
            std::string temp_line;

            std::getline(file, temp_line); // Read x bounds
            std::stringstream ss_x(temp_line);
            ss_x >> xmin >> xmax;

            std::getline(file, temp_line); // Read y bounds
            std::stringstream ss_y(temp_line);
            ss_y >> ymin >> ymax;

            std::getline(file, temp_line); // Read z bounds
            std::stringstream ss_z(temp_line);
            ss_z >> zmin >> zmax;

            this->box_x0[0] = xmin;
            this->box_x0[1] = ymin;
            this->box_x0[2] = zmin;
            this->L[0] = xmax-xmin;
            this->L[1] = ymax-ymin;
            this->L[2] = zmax-zmin;
            state = 0; continue;
        }
        if (line.rfind("ITEM: ATOMS", 0) == 0) { // check if line starts with "ITEM: ATOMS"
            if(line != std::string("ITEM: ATOMS id type xs ys zs")) {
                std::cerr << "Unexpected ATOMS property list: " << line << std::endl;
                return false;
            }
            /*
            std::stringstream ss_header(line);
            std::string temp_item;
            ss_header >> temp_item; // "ITEM:"
            ss_header >> temp_item; // "ATOMS"

            atom_properties.clear();
            std::string property_name;
            while (ss_header >> property_name) {
                atom_properties.push_back(property_name);
            }*/
            this->types.clear();
            this->coords.clear();
            state = 1; continue;
        }
        if(state == 1) { // This is an atom data line
            std::stringstream ss_atom(line);
            Atom atom;
            ss_atom >> atom.id >> atom.type >> atom.x >> atom.y >> atom.z;
            if(atom.id < 1 || atom.id > this->num_atoms) {
                std::cerr << "Invalid atom id: "
                          << atom.id << std::endl;
                return false;
            }

            int idx = atom.id-1;
            this->coords[idx*3+0] = atom.x;//*this->L[0];
            this->coords[idx*3+1] = atom.y;//*this->L[1];
            this->coords[idx*3+2] = atom.z;//*this->L[2];
            this->types[idx] = atom.type;
            natoms++;
            if(natoms == this->num_atoms) {
                return true;
            }
        }
        // else - line is part of an ignored section.
    }

    return false;
}
/*
// Helper function to get a begin iterator
DumpfileIterator begin(std::ifstream& fs) {
    return DumpfileIterator(fs);
}

// Helper function to get an end iterator
DumpfileIterator end(std::ifstream& ) {
    return DumpfileIterator(); // Return default-constructed (end) iterator
}
*/
/* test code
int main() {
    std::string filename("dump.lammpstrj");
    int n = 0;

    for(const Snapshot &snapshot : Dumpfile(filename)) {
        n++;
        std::cout << "First snapshot timestep: " << snapshot.timestep << std::endl;
        std::cout << "Number of atoms in first snapshot: " << snapshot.num_atoms << std::endl;
        std::cout << "Box X bounds: " << snapshot.box_x_min << " " << snapshot.box_x_max << std::endl;
        std::cout << "First atom ID: " << snapshot.atoms[0].id << std::endl;
    }
    std::cout << "Number of snapshots: " << n << std::endl;

    return 0;
}*/
