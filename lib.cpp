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
    SFTensor(std::vector<int> shape, bool allocate=true) {
        this->numDims = shape.size(); this->shape = shape;
        this->stride.resize(numDims); this->offset.resize(numDims);
        for(int dim{numDims-1}; dim >= 0; dim--){ stride[dim] = numElems; numElems *= shape[dim]; }
        if (allocate){ data = (tensorType*) new tensorType[numElems]; isOwner = true; }
        return;
    }

    // Shallow copy constructor
    SFTensor(const SFTensor& tensor){
        this->shape = tensor.shape; this->stride = tensor.stride; this->offset = tensor.offset;
        this->data = tensor.data; this->isOwner = false; this->numDims = tensor.numDims; this->numElems = tensor.numElems;
        return;
    }

    // Returns a new tensor containing the indexed portion of the original tensor but the underlying data is same as original tensor
    SFTensor<tensorType> operator[](int i) const{
        if(numDims == 0) throw runtime_error("Indexing Error: Index out of bounds in operator[i]");
        SFTensor<tensorType> newTensor = SFTensor<tensorType>(*this);
        newTensor.shape.erase(newTensor.shape.begin()); newTensor.stride.erase(newTensor.stride.begin());
        newTensor.offset.erase(newTensor.offset.begin());
        --newTensor.numDims; newTensor.numElems = this->numElems / this->shape[0];
        newTensor.data = this->data + this->stride[0] * (i + this->offset[0]);
        return newTensor;
    }

    // Returns a new tensor containing the indexed portion of the original tensor but the underlying data is same as original tensor
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
    
    // Returns a new tensor containing the indexed portion of the original tensor but the underlying data is same as original tensor
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

    // Returns the LHS tensor and it contains the copy of input data (does not point to the underlying data of input)
    SFTensor<tensorType>& operator=(SFTensor<tensorType> const& input){
        if(this->numDims != input.numDims) throw runtime_error("Dimension mismatch in tensor assignment operator");
        for(int i{0}; i < numDims; i++) if(this->shape[i] != input.shape[i]) throw runtime_error("Shape mismatch between tensor provided to assignment operator");

        _operator_assign(*this, input);
        return *this;
    }

    // Returns the LHS tensor and it contains the copy of input data (does not point the underlying data of input)
    SFTensor<tensorType>& operator=(tensorType input){
        if(this->numDims != 0) throw runtime_error("Broadcasting is not yet supported");
        *data = input;
        return *this;
    }

    // Returns the LHS tensor and it contains the copy of input data (does not point the underlying data of input)
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

    SFTensor<tensorType> operator+(SFTensor<tensorType> const& input){
        if(this->numDims != input.numDims) throw runtime_error("Dimension mismatch in elementwise addition operator");
        for(int i{0}; i < numDims; i++) if(this->shape[i] != input.shape[i]) throw runtime_error("Shape mismatch between tensor provided to elementwise add operator");

        SFTensor<tensorType> result(this->shape);
        _add(*this, input, result);
        return result;
    }

    SFTensor<tensorType> operator-(SFTensor<tensorType> const& input){
        if(this->numDims != input.numDims) throw runtime_error("Dimension mismatch in elementwise subtraction operator");
        for(int i{0}; i < numDims; i++) if(this->shape[i] != input.shape[i]) throw runtime_error("Shape mismatch between tensor provided to elementwise sub operator");

        SFTensor<tensorType> result(this->shape);
        _sub(*this, input, result);
        return result;
    }

    SFTensor<tensorType> operator*(SFTensor<tensorType> const& input){
        if(this->numDims != input.numDims) throw runtime_error("Dimension mismatch in elementwise multiplication operator");
        for(int i{0}; i < numDims; i++) if(this->shape[i] != input.shape[i]) throw runtime_error("Shape mismatch between tensor provided to elementwise mul operator");

        SFTensor<tensorType> result(this->shape);
        _mul(*this, input, result);
        return result;
    }

    SFTensor<tensorType> operator/(SFTensor<tensorType> const& input){
        if(this->numDims != input.numDims) throw runtime_error("Dimension mismatch in elementwise division operator");
        for(int i{0}; i < numDims; i++) if(this->shape[i] != input.shape[i]) throw runtime_error("Shape mismatch between tensor provided to elementwise div operator");

        SFTensor<tensorType> result(this->shape);
        _div(*this, input, result);
        return result;
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

    template<class t>
    friend void _operator_assign(SFTensor<t> first, SFTensor<t> const second);

    template<class t>
    friend void _add(SFTensor<t> const first, SFTensor<t> const second, SFTensor<t> result);

    template<class t>
    friend void _sub(SFTensor<t> const first, SFTensor<t> const second, SFTensor<t> result);

    template<class t>
    friend void _mul(SFTensor<t> const first, SFTensor<t> const second, SFTensor<t> result);

    template<class t>
    friend void _div(SFTensor<t> const first, SFTensor<t> const second, SFTensor<t> result);
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

// Recursive assignment of elements of second tensor to first tensor
template<class t>
void _operator_assign(SFTensor<t> first, SFTensor<t> const second){
        if(first.numDims == 0 && second.numDims == 0){
            *(first.data) = *(second.data);
            return;
        }

        for(int i{0}; i < first.shape[0]; i++) _operator_assign(first[i], second[i]);
        return;
}

// Recursive addition of first tensor to second tensor and storage of result in result tensor
template<class t>
void _add(SFTensor<t> const first, SFTensor<t> const second, SFTensor<t> result){
        if(first.numDims == 0 && second.numDims == 0 && result.numDims == 0){
            *(result.data) = *(first.data) + *(second.data);
            return;
        }

        for(int i{0}; i < first.shape[0]; i++) _add(first[i], second[i], result[i]);
        return;
}

// Recursive subtraction of first tensor - second tensor and storage of result in result tensor
template<class t>
void _sub(SFTensor<t> const first, SFTensor<t> const second, SFTensor<t> result){
        if(first.numDims == 0 && second.numDims == 0 && result.numDims == 0){
            *(result.data) = *(first.data) - *(second.data);
            return;
        }

        for(int i{0}; i < first.shape[0]; i++) _sub(first[i], second[i], result[i]);
        return;
}

// Recursive multiplication of first tensor to second tensor and storage of result in result tensor
template<class t>
void _mul(SFTensor<t> const first, SFTensor<t> const second, SFTensor<t> result){
        if(first.numDims == 0 && second.numDims == 0 && result.numDims == 0){
            *(result.data) = *(first.data) * *(second.data);
            return;
        }

        for(int i{0}; i < first.shape[0]; i++) _mul(first[i], second[i], result[i]);
        return;
}

// Recursive division of first tensor by second tensor and storage of result in result tensor
template<class t>
void _div(SFTensor<t> const first, SFTensor<t> const second, SFTensor<t> result){
        if(first.numDims == 0 && second.numDims == 0 && result.numDims == 0){
            if(*(second.data) == 0) throw runtime_error("Division by 0 error in elementwise division");
            *(result.data) = *(first.data) / *(second.data);
            return;
        }

        for(int i{0}; i < first.shape[0]; i++) _div(first[i], second[i], result[i]);
        return;
}