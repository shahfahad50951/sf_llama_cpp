#include <iostream>
#include "lib.cpp"
using namespace std;

int main(){
    vector<int> t1_shape{2,3};
    SFTensor<double> t1{t1_shape};
    t1.printProperties();
    // t1.rawPrint();
    cout << "here\n";
    SFTensor<double> t2 = t1[1];
    t2.printProperties();
    cout << t1 << '\n';
    cout << t2 << '\n';
    cout << "here2\n";
    return 0;
}