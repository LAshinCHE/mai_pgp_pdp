#include<iostream>
#include<bits/stdc++.h>

using namespace std;

#define BENCHMARK

void PrintVector(vector<double> v){
    for(int i = 0; i < v.size(); i++){
        cout << v[i] << " ";
    }
}

vector<double> MinimumVector(vector<double> v1, vector<double> v2, int n){
    vector<double> minvec(n);
    for(int i = 0; i < n; i++){
        minvec[i] = min(v1[i],v2[i]);
    }
    return minvec;
}

void ReadVector(vector<double>& v){
    for(int i = 0; i < v.size(); i++){
        cin >> v[i];
    }
}

int main(){
    int n;

    cin >> n;

    vector<double> v1(n);
    vector<double> v2(n);

    ReadVector(v1);
    ReadVector(v2);

    #ifdef BENCHMARK
    std::chrono::time_point<std::chrono::system_clock> start, stop;
    start = std::chrono::system_clock::now();
    #endif /* BENCHMARK */
    
    vector<double> minvec = MinimumVector(v1,v2,n);

    #ifdef BENCHMARK
    stop = std::chrono::system_clock::now();
    uint64_t time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    std::cout << time << " us\n";
    #endif /* BENCHMARK */

    #ifndef BENCHMARK
    PrintVector(minvec);
    #endif

    return 0;
}