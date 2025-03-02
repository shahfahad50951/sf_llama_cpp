#include <vector>
#include <ostream>
#include <iostream>
#include <algorithm>
using namespace std;

template <class tensorType>
class SFTensor{
    private:
    int numDims = 0, numElems = 1;
    std::vector<int> shape, stride, offset;
    tensorType* data = nullptr;
    bool isOwner = false;

    public:
    // Construct a new tensor provided an input shape
    SFTensor(std::vector<int> shape, bool allocate=true): numDims{shape.size()}{
        this->shape = shape;
        this->stride.resize(numDims); this->offset.resize(numDims);

        for(int dim{numDims-1}; dim >= 0; dim--){
            stride[dim] = numElems;
            numElems *= shape[dim];
        }

        // for(int i{0}; i < numDims; i++){
        //     cout << numDims-i-1 << "th dimension shape: " << this->shape[i] << '\n';
        //     cout << numDims-i-1 << "th dimension stride: " << this->stride[i] << '\n';
        // }

        if (allocate){
            data = (tensorType*) new tensorType[numElems];
            isOwner = true;
        }
        return;
    }

    // Shallow Copy constructor
    SFTensor(const SFTensor& tensor){
        this->shape = tensor.shape; this->stride = tensor.stride; this->offset = tensor.offset;
        this->data = tensor.data; this->isOwner = false; this->numDims = tensor.numDims;
        this->numElems = tensor.numElems;
        return;
    }

    SFTensor<tensorType> operator[](int i) const{
        if(numDims == 0) throw "Indexing Error: Index out of bounds in operator[]";
        SFTensor<tensorType> newTensor = SFTensor<tensorType>(*this);
        newTensor.shape.erase(newTensor.shape.begin());
        newTensor.stride.erase(newTensor.stride.begin());
        newTensor.offset.erase(newTensor.offset.begin());
        --newTensor.numDims; newTensor.numElems = this->numElems / this->shape[0];
        newTensor.data = this->data + this->stride[0] * (i + this->offset[0]);
        return newTensor;
    }

    // SFTensor operator[](std::vector<int> idx) const{
    //     if(idx.size() > numDims) throw "Index out of Bounds";

    //     int leftDims = numDims - (int)(idx.size());
    //     std::vector<int> newShape;
    //     for(int i{0}; i < leftDims; i++){
    //         newShape.push_back(shape[i]);
    //         // std::cout << "newshape: " << i << ": " << newShape[i] << '\n';
    //     }
    //     int startElemPos = 0;
    //     for(int i{numDims-1}; i >= leftDims; i--){
    //         startElemPos += stride[i] * idx[numDims-1-i];
    //     }
    //     // cout << "startelempos for the view: " << startElemPos << '\n';
    //     // cout << "startelemval for the view: " << data[startElemPos] << '\n';
    //     SFTensor view(newShape, false);
    //     view.data = &data[startElemPos];
    //     return view;
    // }

    void rawPrint(){
        for(int i{0}; i < numElems; i++) std::cout << data[i] << ' ';
        std::cout << '\n';
    }

    void printProperties(){
        cout << "Num Dimensions: " << this->numDims << '\n';
        cout << "Num Elements: " << this->numElems << '\n';
        cout << "Is Owner: " << this->isOwner << '\n';
        for(int i{0}; i < numDims; i++) cout << "Shape: " << shape[i] << "\tStride: " << stride[i] << "\tOffset " << offset[i] << '\n';
        return;
    }

    ~SFTensor(){
        if(isOwner) delete[] data;
    }

    template<class t>
    friend std::ostream& operator<<(std::ostream&, const SFTensor<t>&);

    
};

template<class t>
std::ostream& operator<<(std::ostream& cout, const SFTensor<t>& tensor){
    if(tensor.numDims == 0) cout << *tensor.data;
    else if(tensor.numDims == 1){
        cout << "[ "; for(int i{0}; i < tensor.shape[0]; i++) cout << tensor[i] << ' '; cout << ']';
    }
    else{
        cout << '[';
        for(int i{0}; i < tensor.shape[0]; i++){
            cout << tensor[i];
            if (i == tensor.shape[0]-1) cout << ']'; else cout << '\n';
        }
    }
    return cout;
}