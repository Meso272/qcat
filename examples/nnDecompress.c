#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "ByteToolkit.h"
#include "rw.h"
#include "qcat.h"
#include "zstd.h"
#include <sz_dummy_compression.h>
#define radius 32768

int main(int argc,char*argv[]){
    if(argc < 2)
    {
        printf("Test case: nnDecompress [qpath][upath][ckptpath][outputpath][abseb][dim1][dim2][dim3][actv][global_max][global_min][nmax][nmin]\n");
        
        exit(0);
    }
    int x = 1;
    char *y = (char*)&x;

    if(*y==1)
    {     
        sysEndianType = 0; //LITTLE_ENDIAN_SYSTEM;
        printf("This is little-endian system.\n");
    }
    else //=0
    {
        sysEndianType = 1; //BIG_ENDIAN_SYSTEM;
        printf("This is big-endian system.\n");
    }


    char qpath[640],upath[640],ckptpath[640],outputpath[640];
    sprintf(qpath, "%s", argv[1]);
    sprintf(upath, "%s", argv[2]);
    sprintf(ckptpath, "%s", argv[3]);
    sprintf(outputpath, "%s", argv[4]);
    float abseb=atof(argv[5]);
    int dimx=atoi(argv[6]);
    int dimy=atoi(argv[7]);
    
    int dimz=atoi(argv[8]);
    int actv=atoi(argv[9]);//0:tanh 1:sigmoid 2:no
    float global_max=atof(argv[10]);//0:no norm
    float global_min=atof(argv[11]);
    float norm_max=atof(argv[12]);
    float norm_min=atof(argv[13]);
    


    size_t nbEle,nbEle_ckpt,u_count;
    int status=0;

    int * qs=readInt32Data(qpath,&nbEle,&status);
    
    float * us=readFloatData(upath,&u_count,&status);
    float * ckpts=readFloatData(ckptpath,&nbEle_ckpt,&status);
    float * decompdata=(float *)malloc(nbEle*sizeof(float));
    //printf("%f\n",abseb);
    //float * predictdata=(float *)malloc(nbEle*sizeof(float));
    //printf("%d %d\n",nbEle,nbEle_padded);
    
    int u_idx=0,q_idx=0;
    
    int x_stride=dimy*dimz,y_stride=dimz;
    for(int x=0;x<dimx;x++){
        for(int y=0;y<dimy;y++){
            for(int z=0;z<dimz;z++){
                int cur_idx=x*x_stride+y*y_stride+z;
                float pred;
                if (x&&y&&z){
                    float sum=0;
                    int idx=0;
                    for(int i=x-1;i<=x;i++){
                        for(int j=y-1;j<=y;j++){
                            for(int k=z-1;k<=z;k++){
                                if (i==x&&j==y&&k==z){
                                    sum+=ckpts[idx++];
                                }
                                else{
                                    float val=decompdata[i*x_stride+j*y_stride+k];
                                    if (global_max!=0){
                                        val=(val-global_min)/(global_max-global_min);
                                        val=val*(norm_max-norm_min)+norm_min;
                                    }
                                    sum+=ckpts[idx++]*val;
                                }
                            }
                        }
                    }
                    if (actv==0){
                        
                        sum=tanh(sum);
                        //printf("oritanh%f,",sum);
                        //sum=(exp(sum)-exp(-sum))/(exp(sum)+exp(-sum));
                    }
                    else if (actv==1){
                        sum=1/(1+exp(-sum));
                    }
                    if (global_max!=0){
                        sum=(sum-norm_min)/(norm_max-norm_min);
                        sum=sum*(global_max-global_min)+global_min;
                    }
                    pred=sum;
                    //printf("orig%f,",orig);
                    //printf("pred%f\n",pred);


                }

                else{
                    float f_011=x?decompdata[cur_idx-x_stride]:0;
                    float f_101=y?decompdata[cur_idx-y_stride]:0;
                    float f_110=z?decompdata[cur_idx-1]:0;
                    float f_001=(x&&y)?decompdata[cur_idx-x_stride-y_stride]:0;
                    float f_010=(x&&z)?decompdata[cur_idx-x_stride-1]:0;
                    float f_100=(y&&z)?decompdata[cur_idx-y_stride-1]:0;
                    float f_000=(x&&y&&z)?decompdata[cur_idx-x_stride-y_stride-1]:0;
                    pred=f_000+f_011+f_101+f_110-f_001-f_010-f_100;

                }
                
                //printf("orig%f,",array[cur_idx]);
                //if(x&&y&&z)
                    //printf("initpred:%f\n",pred);

                int q=qs[q_idx++];
               
                if (q==0){
                    pred=us[u_idx++];
                    
                }
                else{
                    q-=radius;
                    if (q<0){
                        q=-q;
                        q<<=1;
                        q=-q;
                    }
                    else{
                        q<<=1;
                    }
                    
                    pred+=q*abseb;
                }
                
                //if(x&&y&&z)
                //printf("finalpred:%f\n",pred);
                 /*
                if (cur_idx==0||cur_idx==262657||cur_idx==263657||cur_idx==264657||cur_idx==265657||cur_idx==300000||cur_idx==400000){
                    printf("curidx: %d\n",cur_idx);
                    printf("q: %d\n",q);
                    printf("pred: %f\n",pred);
                    
                     }
                     */
                decompdata[cur_idx]=pred;
                    
                   
                    
                

            }
        }
    }

    writeFloatData_inBytes(decompdata,nbEle,outputpath,&status);
    return 0;
}