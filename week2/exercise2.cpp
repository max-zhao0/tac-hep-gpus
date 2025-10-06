#include <iostream>

struct Student {
    std::string name;
    std::string email;
    std::string username;
    std::string experiment;
};

void printStudent(const Student& stu) {
    std::cout << "Name: " << stu.name << std::endl;
    std::cout << "Email: " << stu.email << std::endl;
    std::cout << "Username: " << stu.username << std::endl;
    std::cout << "Experiment: " << stu.experiment << std::endl;
}

int main() {
    Student myself;
    myself.name = "Max Zhao";
    myself.email = "max.zhao@princeton.edu";
    myself.username = "zhaom";
    myself.experiment = "CMS";

    printStudent(myself);
}