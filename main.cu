#include <iostream>
#include <fstream>
#include "include/snpreader/SNPReader.h"
#include "include/batchreduce/include/batch_reduce.cuh"
#include "include/batchreduce/include/reduce_functors.cuh"
#include "include/cxxopts/src/cxxopts.hpp"

template <
  typename index_t,
  typename array_t
  >
void convert_data(SNPReader* reader, array_t* cases, index_t sizeCases, array_t* ctrls, index_t sizeCtrls)
{
  vector<SNP*> snps = reader->getSnpSet();

  for(index_t i=0; i < reader->getNumSnp(); i++)
  {
    memcpy(cases+(i*3*sizeCases)+(0*sizeCases), snps[i]->_case0Values, sizeof(array_t)*sizeCases);
    memcpy(cases+(i*3*sizeCases)+(1*sizeCases), snps[i]->_case1Values, sizeof(array_t)*sizeCases);
    memcpy(cases+(i*3*sizeCases)+(2*sizeCases), snps[i]->_case2Values, sizeof(array_t)*sizeCases);

    memcpy(ctrls+(i*3*sizeCtrls)+(0*sizeCtrls), snps[i]->_ctrl0Values, sizeof(array_t)*sizeCtrls);
    memcpy(ctrls+(i*3*sizeCtrls)+(1*sizeCtrls), snps[i]->_ctrl1Values, sizeof(array_t)*sizeCtrls);
    memcpy(ctrls+(i*3*sizeCtrls)+(2*sizeCtrls), snps[i]->_ctrl2Values, sizeof(array_t)*sizeCtrls);
  }
}

int main(int argc, char * argv[]) {
    typedef uint64_t index_t;
    typedef uint32_t array_t;
    // argparser
    cxxopts::Options options(argv[0], "Generate genotype histograms from SNPs");
    options.add_options()
        ("p,tped", "TPED file", cxxopts::value<std::string>(), "FILE")
        ("f,tfam", "TFAM file", cxxopts::value<std::string>(), "FILE")
        ("v,verbose", "Verbosity")
        ("o,out", "Output file", cxxopts::value<std::string>()->default_value("histograms.out")->implicit_value("histograms.out"));
    options.parse(argc, argv);

    // read input files
    SNPReader * reader = new SNPReader(options["tped"].as<std::string>().c_str(), options["tfam"].as<std::string>().c_str());
    reader->loadSNPSet();

    const index_t n = reader->getNumSnp();
    const index_t sizeCases = (reader->getNumCases()+32-1)/32;
    const index_t sizeCtrls = (reader->getNumCtrls()+32-1)/32;

    array_t * cases_h = (array_t*)malloc(sizeof(array_t)*3*sizeCases*n);
    array_t * ctrls_h = (array_t*)malloc(sizeof(array_t)*3*sizeCtrls*n);
    array_t * cases_d; cudaMalloc(&cases_d, sizeof(array_t)*3*sizeCases*n);
    array_t * ctrls_d; cudaMalloc(&ctrls_d, sizeof(array_t)*3*sizeCtrls*n);

    convert_data(reader, cases_h, sizeCases, ctrls_h, sizeCtrls);

    cudaMemcpy(cases_d, cases_h, sizeof(array_t)*3*sizeCases*n, cudaMemcpyHostToDevice);
    cudaMemcpy(ctrls_d, ctrls_h, sizeof(array_t)*3*sizeCtrls*n, cudaMemcpyHostToDevice);

    typedef sum_op_t<array_t> op_t;
    typedef BatchReduce<index_t, array_t, op_t> reduce_t;
    reduce_t reduce = reduce_t();

    array_t * histCases_h = (array_t*)malloc(sizeof(array_t)*3*n);
    array_t * histCtrls_h = (array_t*)malloc(sizeof(array_t)*3*n);
    array_t * histCases_d; cudaMalloc(&histCases_d, sizeof(array_t)*3*n);
    array_t * histCtrls_d; cudaMalloc(&histCtrls_d, sizeof(array_t)*3*n);

    // generate histograms
    reduce(cases_d, sizeCases, 3*n, histCases_d);
    reduce(ctrls_d, sizeCtrls, 3*n, histCtrls_d);

    cudaMemcpy(histCases_h, histCases_d, sizeof(array_t)*3*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(histCtrls_h, histCtrls_d, sizeof(array_t)*3*n, cudaMemcpyDeviceToHost);

    // write histograms to file
    ofstream myfile;
    myfile.open(options["out"].as<std::string>());
    for (index_t i = 0; i < 3*n; i+=3)
    {
        myfile << histCases_h[i] << "\t" << histCases_h[i+1] << "\t" << histCases_h[i+2] << "\t" << histCtrls_h[i] << "\t" << histCtrls_h[i+1] << "\t" << histCtrls_h[i+2] << "\n";
    }
    if(options.count("v"))
    {
        cout << "\ncases\t\t\t|\tctrls\n";
        for (index_t i = 0; i < 3*n; i+=3)
        {
            cout   << histCases_h[i] << "\t" << histCases_h[i+1] << "\t" << histCases_h[i+2] << "\t|\t" << histCtrls_h[i] << "\t" << histCtrls_h[i+1] << "\t" << histCtrls_h[i+2] << "\n";
        }
    }
    myfile.close();

    // free memory
    free(cases_h);
    free(ctrls_h);
    free(histCases_h);
    free(histCtrls_h);
    cudaFree(cases_d);
    cudaFree(ctrls_d);
    cudaFree(histCases_d);
    cudaFree(histCtrls_d);
    delete reader;
}
