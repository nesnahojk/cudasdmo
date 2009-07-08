#include <list>
#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include "cublas.h"
#include <reduce_min_kernel.h>


using namespace std;

void Tokenize(string& str, vector<string>& tokens, const string& delimiters)
{
    string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    string::size_type pos     = str.find_first_of(delimiters, lastPos);
    while (string::npos != pos || string::npos != lastPos)
    {
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = str.find_first_not_of(delimiters, pos);
        pos = str.find_first_of(delimiters, lastPos);
    }
}


__global__ void UpdateH(int j1, int j2,float * k1_device, float * k2_device,float C, float * h_device,int * y_device, int Ntotal)
{

  int t=threadIdx.x;

  int tt = gridDim.x*blockDim.x;
  int ctaStart = blockDim.x*blockIdx.x;
  
  int y_device_j1=y_device[j1];
  int y_device_j2=y_device[j2];

  for(int i=ctaStart+t;i<Ntotal;i+=tt)
  {
    if(i==j1)
    {
        h_device[i]+=y_device[i]*y_device_j1*(k1_device[i]+1.0f/C)+y_device[i]*y_device_j2*k2_device[i];
    }
    else if(i==j2)
    {
        h_device[i]+=y_device[i]*y_device_j1*k1_device[i]+y_device[i]*y_device_j2*(k2_device[i]+1.0f/C);    
    }
    else
    {
        h_device[i]+=y_device[i]*y_device_j1*k1_device[i]+y_device[i]*y_device_j2*k2_device[i];    
    }
  }
}


__global__ void FinishRBFKernel(int which_wanted, float * self_dot_device, int Ntotal,int m, float *cb_dot, float bandwidth)
{

  int t=threadIdx.x;

  int tt = gridDim.x*blockDim.x;
  int ctaStart = blockDim.x*blockIdx.x;
  
  float save=self_dot_device[which_wanted];
  for(int i=ctaStart+t;i<Ntotal;i+=tt)
  {
      cb_dot[i]=expf(-bandwidth*(save+self_dot_device[i]-2*cb_dot[i]));
  }
}


void RBFKernel(int which_wanted,float * x_device, float * self_dot_device, int Ntotal,int m,float *b_device, float bandwidth, int BANDS)
{
    cublasSgemv('t',m,Ntotal,1.0,x_device,m,&x_device[which_wanted*m],1,0.0,b_device,1);

    int gridsize=Ntotal/BANDS;
    dim3 dimBlock(BANDS, 1);
    dim3 dimGrid(gridsize,1);

    FinishRBFKernel<<<dimBlock,dimGrid>>>(which_wanted,self_dot_device,Ntotal,m,b_device,bandwidth);

}



int main(int argc, char *argv[])
{

float bandwidth;
int ITER_MAX;
int BANDS;
float C;
if(argc==5)
{
   ITER_MAX=atoi(argv[1]);
   BANDS=atoi(argv[2]);
   C=atof(argv[3]);
   bandwidth=atof(argv[4]);
}
else
{
    cout<<"Usage: Include the max iteration number and block size"<<endl;
    return EXIT_FAILURE;
}

//START INPUT -----------------------------------------------------------

string s;
int Npos,Nneg,m;

vector<string> t;
getline(cin, s);
Tokenize(s,t,",");
if(t.size()==3)
{
    Nneg=atoi(t[0].c_str());
    Npos=atoi(t[1].c_str());
    m=atoi(t[2].c_str());
}

float *xpos = new float[Npos*m];
float *xneg = new float[Nneg*m];


int cpos=0;
int cneg=0;

while(getline(cin, s)!=NULL)
{

vector<string> t;
Tokenize(s,t," ");

int y=atoi(t[0].c_str());


for(int i=1;i<t.size();i++)
{
    vector<string> t2;
    Tokenize(t[i],t2,":");
    if(y==1)
    {
         xpos[cpos*m+atoi(t2[0].c_str())-1]=atof(t2[1].c_str());
    }
    else
    {
        xneg[cneg*m+atoi(t2[0].c_str())-1]=atof(t2[1].c_str());
    }
}

if(y==1)
{
cpos++;
}
else
{
cneg++;
}


}

int Ntotal=Nneg+Npos;

int * y=new int[Ntotal];
float * h=new float[Ntotal];
float*x=new float[(Npos+Nneg)*m];

for(int i=0;i<Nneg;i++)
{
    h[i]=0;
    y[i]=-1;
    for(int j=0;j<m;j++)
    {
        x[i*m+j]=xneg[i*m+j];
    }
}

for(int i=Nneg;i<Nneg+Npos;i++)
{
    h[i]=0;
    y[i]=1;
    for(int j=0;j<m;j++)
    {
        x[i*m+j]=xpos[(i-Nneg)*m+j];
    }
}


//END INPUT ---------------------------------------------------------------


cublasInit();

int * nu=new int[Ntotal];
float * self_dot=new float[Ntotal];


//move this to CUDA
for(int i=0;i<Ntotal;i++)
{
    self_dot[i]=0;
    nu[i]=0;
    for(int j=0;j<m;j++)
    {
        self_dot[i]+=x[i*m+j]*x[i*m+j];
    }
}




float * x_device;
cublasAlloc(Ntotal*m,sizeof(float),(void**)&x_device);
cublasSetMatrix(m,Ntotal,sizeof(float),x,m,x_device,m);


int * y_device;
cublasAlloc(Ntotal, sizeof(int), (void**)&y_device);
cublasSetVector(Ntotal,sizeof(int),y,1,y_device,1);


float * self_dot_device;
cublasAlloc(Ntotal, sizeof(float), (void**)&self_dot_device);
cublasSetVector(Ntotal,sizeof(float),self_dot,1,self_dot_device,1);


float * h_device;
cublasAlloc(Ntotal, sizeof(float), (void**)&h_device);
cublasSetVector(Ntotal,sizeof(float),h,1,h_device,1);


float * k1_device,*k2_device;
cublasAlloc(Ntotal,sizeof(float),(void**)&k1_device);
cublasAlloc(Ntotal,sizeof(float),(void**)&k2_device);


float *k1=new float[Ntotal];
float *k2=new float[Ntotal];




for(int iter=0;iter<ITER_MAX;iter++)
{
//find min on the two halves of h
int h_neg_min_index=IndexOfMin<float>(h_device,Nneg);
int h_pos_min_index=IndexOfMin<float>(&h_device[Nneg],Npos);


//increase nu for the mins
nu[h_neg_min_index]++;
nu[Nneg+h_pos_min_index]++;

//incrementally maintain the h vector

RBFKernel(h_neg_min_index,x_device, self_dot_device, Ntotal,m,k1_device,bandwidth,BANDS);
RBFKernel(Nneg+h_pos_min_index,x_device, self_dot_device, Ntotal,m,k2_device,bandwidth,BANDS);


int gridsize=Ntotal/BANDS;
dim3 dimBlock(BANDS, 1);
dim3 dimGrid(gridsize,1);
cublasGetVector(Ntotal,sizeof(float),h_device,1,h,1);
//cout<<h_neg_min_index<<" "<<h_pos_min_index+Nneg<<endl;
UpdateH<<<dimBlock,dimGrid>>>(h_neg_min_index,Nneg+h_pos_min_index,k1_device,k2_device,C,h_device,y_device,Ntotal);


}




cublasGetVector(Ntotal,sizeof(float),h_device,1,h,1);


int h_neg_min_index=IndexOfMin<float>(h_device,Nneg);
int h_pos_min_index=IndexOfMin<float>(&h_device[Nneg],Npos);
//int h_neg_min_index=cublasIsamin(Nneg,h_device,1)-1;
//int h_pos_min_index=cublasIsamin(Npos,&h_device[Nneg],1)-1;

//cublasIsamin (int n, const float *x, int incx)

float b=.5*(h[h_pos_min_index+Nneg]-h[h_neg_min_index]);


cout<<Ntotal<<","<<m<<","<<b<<","<<bandwidth<<endl;
for(int i=0;i<Ntotal;i++)
{
    cout<<y[i]<<",";
    cout<<nu[i];

    for(int j=0;j<m-1;j++)
    {
        if(x[i*m+j]!=0)
        {
            cout<<" "<<j+1<<":"<<x[i*m+j];
        }
    }
    cout<<endl;
}
return 0;


}












