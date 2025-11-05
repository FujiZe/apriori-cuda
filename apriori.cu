#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <tuple>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept> 
#include <cstdlib> 
#include <chrono> 
#include <sstream>

using namespace std;
long double round(long double value, int pos){
    long double temp;
    temp = value * pow( 10, pos );
    temp = floor( temp + 0.5 );
    temp *= pow( 10, -pos );
    return temp;
}
#ifndef ITEM_MAX
#define ITEM_MAX 2048
#endif

__constant__ int c_item[ITEM_MAX];

__device__ __forceinline__
bool is_subset_row(const int* __restrict__ flat, int start, int len,
                   const int* __restrict__ item, int m)
{
    if (len < m) return false;
    int i = 0, j = 0;
    const int end = start + len;
    while ((start + i) < end && j < m) {
#if __CUDA_ARCH__
        int a = __ldg(&flat[start + i]);   
#else
        int a = flat[start + i];
#endif
        int b = item[j];
        if (a < b) ++i;
        else if (a > b) return false;      
        else { ++i; ++j; }
    }
    return j == m;
}

__global__
void support_kernel(const int* __restrict__ flat,
                    const int* __restrict__ offs,
                    const int* __restrict__ lens,
                    int num_trans, int m,
                    int* __restrict__ gcount)
{
    extern __shared__ int ssum[];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int local = 0;
    if (tid < num_trans) {
        int start = offs[tid];
        int len   = lens[tid];
        local = is_subset_row(flat, start, len, c_item, m) ? 1 : 0;
    }
    ssum[threadIdx.x] = local;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) ssum[threadIdx.x] += ssum[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(gcount, ssum[0]);
}

class Apriori {
    private:
    int nowStep;
    long double minSupport;
    vector<vector<int> > transactions;
    vector<vector<int>> candidate;
    vector<vector<int>> frequent;
    vector<vector<vector<int> > > frequentSet; 
    vector<tuple<vector<int>, vector<int>, long double, long double> > associationRules;
       
    bool   gpuReady = false;
    int    num_trans = 0;
    size_t nnz = 0;
    int *d_flat = nullptr, *d_offs = nullptr, *d_lens = nullptr, *d_cnt = nullptr;

    void setupGPU() {
        if (gpuReady) return;
        num_trans = (int)transactions.size();

        vector<int> h_flat; h_flat.reserve(1<<20);
        vector<int> h_offs(num_trans), h_lens(num_trans);
        size_t cur = 0;
        for (int t = 0; t < num_trans; ++t) {
            h_offs[t] = (int)cur;
            h_lens[t] = (int)transactions[t].size();
            h_flat.insert(h_flat.end(), transactions[t].begin(), transactions[t].end());
            cur += transactions[t].size();
        }
        nnz = h_flat.size();

        cudaMalloc(&d_flat,  nnz * sizeof(int));
        cudaMalloc(&d_offs,  num_trans * sizeof(int));
        cudaMalloc(&d_lens,  num_trans * sizeof(int));
        cudaMalloc(&d_cnt,   sizeof(int));

        cudaMemcpy(d_flat, h_flat.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_offs, h_offs.data(), num_trans * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lens, h_lens.data(), num_trans * sizeof(int), cudaMemcpyHostToDevice);

        gpuReady = true;
    }

    void releaseGPU() {
        if (d_flat) cudaFree(d_flat), d_flat = nullptr;
        if (d_offs) cudaFree(d_offs), d_offs = nullptr;
        if (d_lens) cudaFree(d_lens), d_lens = nullptr;
        if (d_cnt)  cudaFree(d_cnt),  d_cnt  = nullptr;
        gpuReady = false;
    }

    long double getSupportGPU(const vector<int>& item) {
        if (!gpuReady || num_trans == 0) return 0.0L;
        if (item.empty()) return 0.0L;
        if ((int)item.size() > ITEM_MAX) throw std::runtime_error("item too long");

        cudaMemcpyToSymbol(c_item, item.data(), (int)item.size()*sizeof(int), 0, cudaMemcpyHostToDevice);
        cudaMemset(d_cnt, 0, sizeof(int));

        int block = 256;
        int grid  = (num_trans + block - 1) / block;
        size_t sh = block * sizeof(int);
        support_kernel<<<grid, block, sh>>>(d_flat, d_offs, d_lens, num_trans, (int)item.size(), d_cnt);
        cudaGetLastError();
        cudaDeviceSynchronize();

        int h = 0;
       cudaMemcpy(&h, d_cnt, sizeof(int), cudaMemcpyDeviceToHost);
        return (long double)h / (long double)num_trans * 100.0L;
    }
    public:
     Apriori (vector<vector<int> > _transactions, long double _minSupport) {
        nowStep=0;
        minSupport = _minSupport;
        for(auto&row:_transactions){
            sort(row.begin(), row.end());
            transactions.push_back(row);
        }
        frequentSet.push_back({{}});
    }
    ~Apriori() { releaseGPU(); }
    vector<tuple<vector<int>, vector<int>, long double, long double> > getAssociationRules(){
        return associationRules;
    }
    void process() {
        while(true) {
            candidate = generate_candidate();
            if(candidate.size()==0) break;
            nowStep++;
            
            frequent = generate_frequent();
            frequentSet.push_back(frequent);
        }
        
        for(auto&stepItemSet:frequentSet) {
            for(auto&items:stepItemSet) {
                generateAssociationRule(items, {}, {}, 0);
            }
        }
    }
    vector<vector<int>> generate_candidate(){
        if(nowStep == 0){
            vector<vector<int> > ret;
            vector<int> element;
            set<int> s;
            for(auto&row:transactions){ 
                for(auto&col:row)
                    s.insert(col);
            }
            for (int x : s)
                element.push_back(x);
            for (auto &i : element)
                ret.push_back(vector<int>(1, i));
            return ret;
        }
        else {
                return pruning(joining());
            }
    }
    vector<vector<int> > generate_frequent() {
        vector<vector<int> > ret;
        for(auto&row:candidate){
            long double support = getSupport(row);
            if(round(support, 2) < minSupport) continue;
            ret.push_back(row);
        }
        return ret;
    }
    void generateAssociationRule(vector<int> items, vector<int> X, vector<int> Y, int index) {
        if(index == items.size()) {
            if(X.size()==0 || Y.size() == 0) return;
            long double XYsupport = getSupport(items);
            long double Xsupport = getSupport(X);
            
            if(Xsupport == 0) return;
            
            long double support = (long double)XYsupport;
            long double confidence = (long double)XYsupport/Xsupport*100.0;
            associationRules.push_back({X, Y, support, confidence});
            return;
        }
        
        X.push_back(items[index]);
        generateAssociationRule(items, X, Y, index+1);
        X.pop_back();
        Y.push_back(items[index]);
        generateAssociationRule(items, X, Y, index+1);
    }
    vector<vector<int> > joining () {
        vector<vector<int> > ret;
        for(int i=0;i<frequent.size();i++){
            for(int j=i+1;j<frequent.size(); j++) {
                int k;
                for(k=0;k<nowStep-1; k++) {
                    if(frequent[i][k] != frequent[j][k]) break;
                }
                if(k == nowStep-1) {
                    vector<int> tmp;
                    for(int k=0;k<nowStep-1; k++) {
                        tmp.push_back(frequent[i][k]);
                    }
                    int a = frequent[i][nowStep-1];
                    int b = frequent[j][nowStep-1];
                    if(a>b) swap(a,b);
                    tmp.push_back(a), tmp.push_back(b);
                    ret.push_back(tmp);
                }
            }
        }
        return ret;
    }
     vector<vector<int> > pruning (vector<vector<int> > joined) {
        vector<vector<int> > ret;
        
        set<vector<int> > lSet;
        for(auto&row:frequent) lSet.insert(row);
        
        for(auto&row:joined){
            int i;
            for(i=0;i<row.size();i++){
                vector<int> tmp = row;
                tmp.erase(tmp.begin()+i);
                if(lSet.find(tmp) == lSet.end()) {
                    break;
                }
            }
            if(i==row.size()){
                ret.push_back(row);
            }
        }
        return ret;
    }
long double getSupport(vector<int> item) {
        sort(item.begin(), item.end());
#ifdef USE_CUDA
        try {
            setupGPU();
            return getSupportGPU(item);
        } catch (const std::exception&) {}
#endif
        int cnt = 0;
        for (auto& row : transactions) {
            if (row.size() < item.size()) continue;
            int i = 0, j = 0;
            while (i < (int)row.size() && j < (int)item.size()) {
                if (row[i] < item[j]) ++i;
                else if (row[i] > item[j]) break;
                else { ++i; ++j; }
            }
            if (j == (int)item.size()) cnt++;
        }
        return (long double)cnt / (long double)transactions.size() * 100.0L;
    }
};
class InputReader {
private:
    ifstream fin;
    vector<vector<int> > transactions;
public:
    InputReader(string filename) {
        fin.open(filename);
        if(!fin) {
            cout << "Input file could not be opened\n";
            exit(0);
        }
        parse();
    }
    void parse() {
    string line;
    while (std::getline(fin, line)) {
     
        bool has_digit = false;
        for (char c : line) { if (std::isdigit((unsigned char)c)) { has_digit = true; break; } }
        if (!has_digit) continue;

        std::istringstream ss(line);
        std::vector<int> arr;
        int x;
        while (ss >> x) arr.push_back(x);
        if (arr.empty()) continue;

        std::sort(arr.begin(), arr.end());
        arr.erase(std::unique(arr.begin(), arr.end()), arr.end());
        transactions.push_back(std::move(arr));
    }
}
    vector<vector<int> > getTransactions() {
        return transactions;
    }
};

class OutputPrinter {
private:
    ofstream fout;
    vector<tuple<vector<int>, vector<int>, long double, long double> > associationRules;
public:
    OutputPrinter(string filename, vector<tuple<vector<int>, vector<int>, long double, long double> > _associationRules) {
        fout.open(filename);
        if(!fout) {
            cout << "Ouput file could not be opened\n";
            exit(0);
        }
        associationRules = _associationRules;
        buildOutput();
    }
    
    void buildOutput() {
        for(auto&i:associationRules) {
            fout << vectorToString(get<0>(i)) << '\t';
            fout << vectorToString(get<1>(i)) << '\t';
            
            fout << fixed;
            fout.precision(2);
            fout << get<2>(i) << '\t';
            
            fout << fixed;
            fout.precision(2);
            fout << get<3>(i);
            
            fout << endl;
        }
    }
    
    string vectorToString(vector<int> arr) {
        string ret = "{";
        for(int i=0;i<arr.size();i++){
            ret += to_string(arr[i]);
            if(i != arr.size()-1){
                ret += ",";
            }
        }
        ret += "}";
        return ret;
    }
};
int main (int argc, char ** argv) {
    if(argc!=4) {
        cout << "error : The number of parameters must be 3";
        return 0;
    }
    string minSupport(argv[1]);
    string inputFileName(argv[2]);
    string outputFileName(argv[3]);

    
    InputReader inputReader(inputFileName);
    vector<vector<int> > transactions = inputReader.getTransactions();
    
    Apriori apriori(transactions, stold(minSupport));
    auto start_total = chrono::high_resolution_clock::now();
    apriori.process();
     auto end_total = chrono::high_resolution_clock::now();
     double total_ms = chrono::duration<double, milli>(end_total - start_total).count();
    OutputPrinter outputPrinter(outputFileName, apriori.getAssociationRules());


    
    cout << fixed << setprecision(3);
    cout << "\n================ Execution Summary ================\n";
#ifdef USE_CUDA
    cout << "GPU (CUDA) support acceleration enabled.\n";
#else
    cout << "CPU-only mode (no CUDA acceleration).\n";
#endif
    cout << "Total Apriori time: " << total_ms << " ms (" << total_ms/1000.0 << " s)\n";
    cout << "===================================================\n";

    return 0;
}