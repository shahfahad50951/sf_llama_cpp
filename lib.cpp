#include <vector>
#include <ostream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
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

        if (allocate){
            data = (tensorType*) new tensorType[numElems];
            isOwner = true;
        }
        return;
    }

    // Shallow copy constructor
    SFTensor(const SFTensor& tensor){
        this->shape = tensor.shape; this->stride = tensor.stride; this->offset = tensor.offset;
        this->data = tensor.data; this->isOwner = false; this->numDims = tensor.numDims;
        this->numElems = tensor.numElems;
        return;
    }

    SFTensor<tensorType> operator[](int i) const{
        if(numDims == 0) throw runtime_error("Indexing Error: Index out of bounds in operator[i]");
        SFTensor<tensorType> newTensor = SFTensor<tensorType>(*this);
        newTensor.shape.erase(newTensor.shape.begin());
        newTensor.stride.erase(newTensor.stride.begin());
        newTensor.offset.erase(newTensor.offset.begin());
        --newTensor.numDims; newTensor.numElems = this->numElems / this->shape[0];
        newTensor.data = this->data + this->stride[0] * (i + this->offset[0]);
        return newTensor;
    }

    SFTensor<tensorType> operator[](pair<int,int> slice){
        int i = slice.first, j = slice.second;
        if(numDims == 0) throw runtime_error("Indexing Error: Index out of bounds in operator[i,j]");
        if(j <= i) throw runtime_error("Indexing Error: End Index <= Start Index");
        SFTensor<tensorType> newTensor{*this};
        newTensor.shape[0] = j-i;
        newTensor.offset[0] += i;
        int perOuterDimElems = this->numElems / this->shape[0];
        newTensor.numElems = perOuterDimElems * newTensor.shape[0]; 
        return newTensor;
    }
    
    SFTensor<tensorType> operator[](vector<pair<int,int>> slices){
        int totalSliceDims = slices.size();
        if(numDims < totalSliceDims) throw runtime_error("Indexing Error: Index out of bounds in operator[i1:j1, i2:j2, ...]");
        SFTensor<tensorType> newTensor{*this};
        for(int dim{0}; dim < totalSliceDims; dim++){
            int i = slices[dim].first, j = slices[dim].second;
            if(j <= i) throw runtime_error("Indexing Error: End Index <= Start Index");
            newTensor.shape[dim] = j-i;
            newTensor.offset[dim] += i;
        }
        newTensor.numElems = 1;
        for(int num: newTensor.shape) newTensor.numElems *= num;
        return newTensor;
    }

    SFTensor<tensorType>& operator=(tensorType input){
        if(this->numDims != 0) throw runtime_error("Broadcasting not yet supported");
        *data = input;
        return *this;
    }

    SFTensor<tensorType>& operator=(vector<tensorType> const& input){
        if(this->numDims != 1) throw runtime_error("Broadcasting not yet supported");
        int inputSz = input.size();
        if(this->shape[0] != inputSz) throw runtime_error("Mismatch in tensor and vector shape");
        for(int i{0}; i < inputSz; i++){
            *this[i] = input[i];
        }
        return *this;
    }

    SFTensor<tensorType>& operator=(vector<vector<tensorType>> const& input){
        if(this->numDims != 2) throw runtime_error("Broadcasting not yet supported");
        int inputSz0 = input.size(), inputSz1 = input[0].size();
        if((this->shape[0] != inputSz0) || (this->shape[1] != inputSz1)) throw runtime_error("Mismatch in tensor and vector shape");
        for(int i{0}; i < inputSz0; i++){
            for(int j{0}; j < inputSz1; j++){
                (*this)[i][j] = input[i][j];
            }
        }
        return *this;
    }

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