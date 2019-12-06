
//Bibliotecas Basicas
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <random>
#include <fstream>
#include <cstdlib>
#include <time.h>

//Biblioteca Thrust
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>


//Biblioteca cuRAND
#include <curand.h>
#include <curand_kernel.h>


//PARAMETROS GLOBAIS
const int QUANT_PAIS_AVALIA = 4;
int POP_TAM = 200;
int N_CIDADES = 20;
int BLOCKSIZE = 1024;
int TOTALTHREADS = 2048;
int N_GERA = 100;
const int MUT = 10;
const int MAX = 19;
const int MIN = 0;
const int ELITE = 2;

/*
 * Busca por erros nos processos da gpu
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/*
 * Calcula a distacia total representada por permutações de cidades de cada indivíduo
 * Utiliza uma matriz M de distancias carregada de um arquivo binario na main
 */
__global__ void fitness(unsigned int n, unsigned int np, float*M, int *V, double *fitness) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        double value;

        if (index < n) {
                for (int i=index; i < n; i += stride) {
                        value = 0;
                        for (int p=0; p<(np-1); p++){
                                value += M[V[i*np+p]*np+V[i*np+p+1]];
                        }
                        fitness[i] = value;
                }
        }
}

/*
 * Gera uma seleção(pool) de pais para cruzar utilizando método de torneio
 */
__global__ void escolhePais(unsigned int n, unsigned int np, int *paisAle, double *fitness, int *pool) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;


        for (int i=index; i<n; i+=stride) {
                double best = 10000.0;
                int best_index = -1;
                int idx;

                for (int j=0; j<QUANT_PAIS_AVALIA; j++) {
                        idx = paisAle[i*QUANT_PAIS_AVALIA+j];
                        if (fitness[idx] < best) {
                                best = fitness[idx];
                                best_index = idx;
                        }
                }
                pool[i] = best_index;
        }
}

/*
 * A partir da geração atual gera uma nova utilizando os operadores de cruzamento(multi point crossover) e mutação
 * Os parametros de probabilidade do evento de elitismo é dado por ELITE, e o parametro de mutação é MUT
 */
__global__ void cruza(unsigned int n, unsigned int np, int *cidadesAle, int *pop, int *newPop, int *poolPais, int *mutacoes) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        int paiA, paiB, copiaPai, crossover, mutar, pontoMutar;

        for (int i=index; i<n; i+=stride) {
                copiaPai = cidadesAle[i*4];
                crossover = cidadesAle[(i+1)*4] % np;
                mutar = cidadesAle[(i+2)*4];
                pontoMutar = cidadesAle[(i+3)*4] % np;
                paiA = poolPais[i];
                paiB = poolPais[i+1];

                if (copiaPai < ELITE) {
                        for (int j=0; j<np; j++) {
                                newPop[(i*np) + j] = pop[(paiA*np) + j];
                                continue;
                        }
                }
                for(int j=0;j<np;j++)
                {
                	newPop[(i*np) + j] = pop[(paiA*np) + j];
                }
                int t=0, aux=0, crossoverSup;
                crossoverSup=(crossover +mutacoes[i]>MAX)?(MAX):(crossover +mutacoes[i]);
                for(int j=crossover; j<crossoverSup;j++)
                {
                	t=0;
                    while(newPop[(i*np) +t]!=pop[(paiB*np) + j])
                    {
                    	t++;
                    }
                    aux = newPop[i*np+j];
                    newPop[i*np+j] = newPop[i*np+t];
                    newPop[i*np+t] = aux;

                }

                if (mutar < MUT) {
                	int mut = (mutacoes[i]>MAX)?(MAX):((mutacoes[i]<MIN)?(MIN):(mutacoes[i]));
                	t=0;
                	while(newPop[(i*np) +t]!=mut)
                	{
                		t++;
                	}
                	aux = newPop[i*np+pontoMutar];
                	newPop[i*np+pontoMutar] = newPop[i*np+t];
                	newPop[i*np+t] = aux;

                }

        }

}

/*
 * Preenche um vetor com a sequencia de genes possiveis repitidos n vezes
 */
__global__ void preencheGenes(unsigned int n,unsigned int np, int* genes)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i=index; i<n; i+=stride)
		for(int j=0;j<np;j++)
			genes[i*np+j]=j;

}

/*
 * Gera n indivíduos para uma população inicial respeitando as condições do problema, no caso um TSP
 */
__global__ void popInicial(unsigned int n,unsigned int np,int* v, int* genes, int* ale)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
		for (int i=index; i<n; i+=stride)
		{
			for(int j=0; j<np; j++)
			                {
			                    int p = (ale[i*np+j]<j)?j:ale[i*np+j];
			                    v[i*np+j] = genes[i*np+p];
			                    int aux = genes[i*np+j];
			                    genes[i*np +j] = genes[i*np+p];
			                    genes[i*np+p]=aux;
			                }
		}
}

/*
 * Realiza o comando init nos estados na curand utilizando uma seed
 */
__global__ void init(unsigned int seed, curandState_t* states) {
        int idx = threadIdx.x+blockDim.x*blockIdx.x;

        curand_init(seed, idx, 0,  &states[idx]);
}

/*
 * Gera um vetor de numeros aleatorios menores que max utilizando os estados curand já iniciados
 */
__global__ void setRandom(curandState_t* states, int* numbers, int max) {
        int idx = threadIdx.x+blockDim.x*blockIdx.x;

                numbers[idx] = curand(&states[idx]) % max;

}

int main() {
        int generacao = 0; //Contador de geracoes

        //Vetores do Host
        int* pop = new int[POP_TAM * N_CIDADES];
        double* popFitness= new double[POP_TAM];
        int* newPop= new int[POP_TAM * N_CIDADES];
        float* matriz= new float[N_CIDADES*N_CIDADES];
        int* ale = new int[POP_TAM*N_CIDADES];

        //Carrega um arquivo binario com os dados das distancias
        std::ifstream fs;
        fs.open("mat.dat", std::ios::binary);
        float m;
        for(int i=0;i<N_CIDADES*N_CIDADES;i++){
        	fs.seekg(i*sizeof(float), std::ios::beg);
            fs.read((char*)&m, (sizeof(float)));
            matriz[i]=m;
        }
        fs.close();

        //Criação e alocacão dos vetores na gpu
        int* mutPtr;
        gpuErrchk(cudaMalloc((void **)&mutPtr, POP_TAM*sizeof(int)));
        int* popPtr;
        gpuErrchk(cudaMalloc((void **)&popPtr, N_CIDADES*POP_TAM*sizeof(int)));
        int* newPopPtr;
        gpuErrchk(cudaMalloc((void **)&newPopPtr, N_CIDADES*POP_TAM*sizeof(int)));
        double* fitnessPtr;
        gpuErrchk(cudaMalloc((void **)&fitnessPtr, POP_TAM*sizeof(double)));
        float* mPtr;
        gpuErrchk(cudaMalloc((void **)&mPtr, N_CIDADES*N_CIDADES*sizeof(float)));
        int* genesPtr;
        gpuErrchk(cudaMalloc((void**)&genesPtr, POP_TAM*N_CIDADES*sizeof(int)));
        int* alePtr;
        gpuErrchk(cudaMalloc((void**)&alePtr, POP_TAM*N_CIDADES*sizeof(int)));
        int *paisAle;
        gpuErrchk(cudaMalloc((void**) &paisAle, POP_TAM*QUANT_PAIS_AVALIA*sizeof(int)));
        int *poolPais_d;
        gpuErrchk(cudaMalloc((void**) &poolPais_d, (POP_TAM+1)*sizeof(int)));
        int *cidadesAle_d;
        gpuErrchk(cudaMalloc((void**)&cidadesAle_d, (POP_TAM+3)*4*sizeof(int)));

        //Criação e alocação dos vetores de estados na gpu
        curandState_t* states;
        gpuErrchk(cudaMalloc((void**) &states, POP_TAM*QUANT_PAIS_AVALIA*sizeof(curandState_t)));
        curandState_t* mutat;
        gpuErrchk(cudaMalloc((void**) &mutat, POP_TAM*QUANT_PAIS_AVALIA*sizeof(curandState_t)));
        curandState_t* initStates;
        gpuErrchk(cudaMalloc((void**) &initStates, POP_TAM*N_CIDADES*sizeof(curandState_t)));
        curandState_t* filhosStates;
        gpuErrchk(cudaMalloc((void**) &filhosStates,(POP_TAM+3)*4*sizeof(curandState_t)));


        //Verifica as capacidades do gpu instalada e compara com as capacidades pedidas
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Device only supports a BLOCKSIZE of " << prop.maxThreadsPerBlock <<" threads/block" << std::endl;
        if (BLOCKSIZE > prop.maxThreadsPerBlock) {
                std::cout << "Device only supports a BLOCKSIZE of " << prop.maxThreadsPerBlock <<" threads/block" << std::endl;
                std::cout << "Try again with a smaller BLOCKSIZE" << std::endl;
                return 1;
        }
        std::cout << "Running with " << TOTALTHREADS << " threads and a BLOCKSIZE of ";
        std::cout << BLOCKSIZE << std::endl;
        int numBlocks = TOTALTHREADS/BLOCKSIZE;
        if (TOTALTHREADS % BLOCKSIZE != 0) {
                ++numBlocks;
                TOTALTHREADS = numBlocks*BLOCKSIZE;
        }
        std::cout << "Running " << numBlocks << " total blocks." << std::endl;



        //Preenche o vetor Genes
        preencheGenes<<<TOTALTHREADS,BLOCKSIZE>>>(POP_TAM, N_CIDADES, genesPtr);
        gpuErrchk(cudaPeekAtLastError());

        //Inicializa parametros aleatorios para a pop inicial
        init<<<POP_TAM*N_CIDADES, 1>>>(time(0), initStates);
        gpuErrchk(cudaPeekAtLastError());

        //Gera parametros aleatorios para a pop inicial
        setRandom<<<POP_TAM*N_CIDADES, 1>>>(initStates, alePtr, N_CIDADES);
        gpuErrchk(cudaPeekAtLastError());

        //Libera a menoria dos estados curand usados para gerar os parametros iniciais
        cudaFree(initStates);

        //Gera a população inicial
        popInicial<<<TOTALTHREADS,BLOCKSIZE>>>(POP_TAM,N_CIDADES,popPtr, genesPtr, alePtr);
        gpuErrchk(cudaPeekAtLastError());

        //Copia a pop inicial para o host
        cudaMemcpy(pop, popPtr, POP_TAM*N_CIDADES*sizeof(int), cudaMemcpyDeviceToHost);
        gpuErrchk(cudaPeekAtLastError());

        //libera a memoria dos parametros iniciais e do vetor de genes da gpu
        cudaFree(alePtr);
        cudaFree(genesPtr);

        //Copia a matriz de distancias para a gpu
        cudaMemcpy(mPtr, matriz, N_CIDADES*N_CIDADES*sizeof(float), cudaMemcpyHostToDevice);
        gpuErrchk(cudaPeekAtLastError());

        //Calcula a fitness de toda população
        fitness<<<TOTALTHREADS, BLOCKSIZE>>>(POP_TAM, N_CIDADES, mPtr, popPtr, fitnessPtr);

        //Copia as fitness para o host
        cudaMemcpy(popFitness, fitnessPtr, POP_TAM*sizeof(double), cudaMemcpyDeviceToHost);
        gpuErrchk(cudaPeekAtLastError());

        //Encontra e printa o melhor individuo
        int i_melhor = thrust::min_element(popFitness, popFitness + POP_TAM-1) - popFitness;
        double melhor =popFitness[i_melhor];
        std::cout << "Geracao inicial, melhor fitness: " << melhor << " no indice: " << i_melhor << "   ";
        for (int i=0; i<N_CIDADES; i++) {
                std::cout << pop[i_melhor * N_CIDADES + i] << " ";
        }
        std::cout << "\n";


        //Inicia a contagem de tempo
        clock_t Ticks[2];
        Ticks[0] = clock();

        //Loop principal
        for(int r=0;r<N_GERA;r++) {

        		//Inicia e gera indices de pais aleatorios
                init<<<POP_TAM*QUANT_PAIS_AVALIA, 1>>>(time(0), states);
                gpuErrchk(cudaPeekAtLastError());
                setRandom<<<POP_TAM*QUANT_PAIS_AVALIA, 1>>>(states, paisAle, POP_TAM);
                gpuErrchk(cudaPeekAtLastError());

                //Escolhe uma seleção de pais
                escolhePais<<<TOTALTHREADS, BLOCKSIZE>>>(POP_TAM, N_CIDADES, paisAle, fitnessPtr, poolPais_d);
                gpuErrchk(cudaPeekAtLastError());

                //inicia e gera parametros aleatorios para serem usados no cruzamento
                init<<<(POP_TAM+3)*4, 1>>>(time(0), filhosStates);
                gpuErrchk(cudaPeekAtLastError());
                setRandom<<<(POP_TAM+3)*4, 1>>>(filhosStates, cidadesAle_d, 100);
                gpuErrchk(cudaPeekAtLastError());

                //Inicia e gera mutações aleatorias
                init<<<POP_TAM, 1>>>(time(0), mutat);
                gpuErrchk(cudaPeekAtLastError());
                setRandom<<<POP_TAM, 1>>>(mutat, mutPtr, N_CIDADES);
                gpuErrchk(cudaPeekAtLastError());

                //cruza a população e armazena nda newPopPtr
                cruza<<<TOTALTHREADS, BLOCKSIZE>>>(POP_TAM, N_CIDADES, cidadesAle_d, popPtr, newPopPtr, poolPais_d, mutPtr);
                gpuErrchk(cudaPeekAtLastError());

                //Copia a nova pop para o host
                cudaMemcpy(pop, newPopPtr, N_CIDADES*POP_TAM*sizeof(int), cudaMemcpyDeviceToHost);
                gpuErrchk(cudaPeekAtLastError());


                //Copia a nova pop para o device
                cudaMemcpy(popPtr, pop, N_CIDADES*POP_TAM*sizeof(int), cudaMemcpyHostToDevice);
                gpuErrchk(cudaPeekAtLastError());

                //Avalia cada individuo
                fitness<<<TOTALTHREADS, BLOCKSIZE>>>(POP_TAM, N_CIDADES,mPtr,  popPtr, fitnessPtr);
                gpuErrchk(cudaPeekAtLastError());

                //Copia o vetor de fitness para o host
                cudaMemcpy(popFitness, fitnessPtr, POP_TAM*sizeof(double), cudaMemcpyDeviceToHost);
                gpuErrchk(cudaPeekAtLastError());

                //Mostra o melhor
                i_melhor = thrust::min_element(popFitness, popFitness + POP_TAM-1) - popFitness;
                melhor =popFitness[i_melhor];
                std::cout << "Geracao: " << generacao << " Melhor Fitness: " << melhor << " no indice: " << i_melhor << "   ";
                for (int i=0; i<N_CIDADES; i++) {
                        std::cout << pop[i_melhor * N_CIDADES + i] << " ";
                }
                std::cout << std::endl;

                //Soma no contador de gerações
                generacao++;
        }
        //finaliza a contagem de tempo
        Ticks[1] = clock();
        double Tempo = ((Ticks[1] - Ticks[0]) * 1000.0 / CLOCKS_PER_SEC)/N_GERA;
        std::cout<<"\nTempo: "<<Tempo;

        //Libera memoria
        cudaFree(states);
        cudaFree(paisAle);
        cudaFree(poolPais_d);
        cudaFree(filhosStates);
        cudaFree(cidadesAle_d);
        cudaFree(fitnessPtr);
        cudaFree(popPtr);
        cudaFree(mPtr);
        cudaFree(newPopPtr);
        cudaFree(mutPtr);
        cudaFree(mutat);



        return 0;
}
