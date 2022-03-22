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


inline int quantize_api(float *data, float pred, float error_bound) {
    float data_value=*data;
    float diff = data_value - pred;
    int quant_index = (int) (fabs(diff) / error_bound) + 1;
    if (quant_index < radius * 2) {
        quant_index >>= 1;
        int half_index = quant_index;
        quant_index <<= 1;
        int quant_index_shifted;
        if (diff < 0) {
            quant_index = -quant_index;
            quant_index_shifted = radius - half_index;
        } else {
            quant_index_shifted = radius + half_index;
        }
        float decompressed_data = pred + quant_index * error_bound;
        if (fabs(decompressed_data - data_value) > error_bound) {
            return 0;
        } else {
            *data = decompressed_data;//??
            return quant_index_shifted;
        }
    } else {
        return 0;
    }
}
/*
float lorenzo_2d(float * array, int x_start,int y_start, float error_bound, 
    int x_size, int y_size, int block_size, int * qs, float * us, int *u_count, float * preds)//eb is absolute
{
    int x_end=x_start+block_size;
    if (x_end>x_size) x_end=x_size;
    int y_end=y_start+block_size;
    if (y_end>y_size) y_end=y_size;
    float loss=0;
    int qidx=0,uidx=0;
    for (int x=x_start;x<x_end;x++){
        for(int y=y_start;y<y_end;y++){
            int idx=x*y_size+y;
            int idx_depadded=(x-1)*(y_size-1)+y-1;
            float orig=array[idx];
            float a=array[idx-y_size];
            float b=array[idx-1];
            float c=array[idx-y_size-1];
            float pred=a+b-c;
            preds[idx_depadded]=pred;
            loss+=fabs(orig-pred);
            int q=quantize_api(&orig,pred,error_bound);
            qs[qidx++]=q;
            if (q==0)
                us[uidx++]=orig;
            array[idx]=orig;


        }
    }
    (*u_count)=uidx;
    return loss;
}

float lorenzo_3d(float * array, int x_start,int y_start, int z_start,float error_bound, 
    int x_size, int y_size, int z_size,int block_size, int * qs, float * us, int *u_count, float * preds)//eb is absolute
{
    int x_end=x_start+block_size;
    if (x_end>x_size) x_end=x_size;
    int y_end=y_start+block_size;
    if (y_end>y_size) y_end=y_size;
    int z_end=z_start+block_size;
    if (z_end>z_size) z_end=z_size;
    float loss=0;
    int qidx=0,uidx=0;
    int x_stride=y_size*z_size,y_stride=z_size;
    int x_stride_depadded=(y_size-1)*(z_size-1);
    for (int x=x_start;x<x_end;x++){
        for(int y=y_start;y<y_end;y++){
            for(int z=z_start;z<z_end;z++){
                
                int idx=x*x_stride+y*y_stride+z;
                int idx_depadded=(x-1)*(x_stride_depadded)+(y-1)*(y_stride-1)+(z-1);
                float orig=array[idx];
                float f_011=array[idx-x_stride];
                float f_101=array[idx-y_stride];
                float f_110=array[idx-1];
                float f_001=array[idx-x_stride-y_stride];
                float f_010=array[idx-x_stride-1];
                float f_100=array[idx-y_stride-1];
                float f_000=array[idx-x_stride-y_stride-1];
                float pred=f_000+f_011+f_101+f_110-f_001-f_010-f_100;
                preds[idx_depadded]=pred;
                loss+=fabs(orig-pred);
                int q=quantize_api(&orig,pred,error_bound);
                qs[qidx++]=q;
                if (q==0)
                    us[uidx++]=orig;
                array[idx]=orig;


            }
        }
    }

    (*u_count)=uidx;
    return loss;
}
*/
int main(int argc,char*argv[]){
    if(argc < 2)
    {
        printf("Test case: nnPredict [inputpath][ckptpath][vrrel error_bound][dim1][dim2][dim3][actv][global_max][global_min][nmax][nmin]\n");
        
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


    char inputpath[640],ckptpath[640];
    sprintf(inputpath, "%s", argv[1]);
    sprintf(ckptpath, "%s", argv[2]);
    float eb=atof(argv[3]);
    int dimx=atoi(argv[4]);
    int dimy=atoi(argv[5]);
    
    int dimz=atoi(argv[6]);
    int actv=atoi(argv[7]);//0:tanh 1:sigmoid 2:no
    float global_max=atof(argv[8]); //0:no norm
    float global_min=atof(argv[9]);
    float norm_max=atof(argv[10]);
    float norm_min=atof(argv[11]);


    size_t nbEle,nbEle_ckpt;
    int status=0;

    float * array=readFloatData(inputpath,&nbEle,&status);
    
    float * ckpts=readFloatData(ckptpath,&nbEle_ckpt,&status);
    //float * decompdata=(float *)malloc(nbEle*sizeof(float));
    //float * predictdata=(float *)malloc(nbEle*sizeof(float));
    //printf("%d %d\n",nbEle,nbEle_padded);
    float min=2e30;
    float max=-2e30;
    
    for(int x=0;x<dimx;x++){
        for (int y=0;y<dimy;y++){
            for(int z=0;z<dimz;z++){
                int idx=x*dimy*dimz+y*dimz+z;
                float ele=array[idx];
                max=ele>max?ele:max;
                min=ele<min?ele:min;
            }


        }
    }

    
    float range=max-min;
    
    float abseb=eb*range;
    printf("%.24f\n",abseb);
    
    int * final_qs=(int*)malloc(nbEle*sizeof(int));
    float * final_us=(float*)malloc(nbEle*sizeof(float));
    
    

    int x_stride=dimy*dimz,y_stride=dimz,q_count=0,u_count=0;
    for(int x=0;x<dimx;x++){
        for(int y=0;y<dimy;y++){
            for(int z=0;z<dimz;z++){
                int cur_idx=x*x_stride+y*y_stride+z;
                float pred;
                float orig=array[cur_idx];
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
                                    float val=array[i*x_stride+j*y_stride+k];
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
                    float f_011=x?array[cur_idx-x_stride]:0;
                    float f_101=y?array[cur_idx-y_stride]:0;
                    float f_110=z?array[cur_idx-1]:0;
                    float f_001=(x&&y)?array[cur_idx-x_stride-y_stride]:0;
                    float f_010=(x&&z)?array[cur_idx-x_stride-1]:0;
                    float f_100=(y&&z)?array[cur_idx-y_stride-1]:0;
                    float f_000=(x&&y&&z)?array[cur_idx-x_stride-y_stride-1]:0;
                    pred=f_000+f_011+f_101+f_110-f_001-f_010-f_100;

                }
                
                //printf("orig%f,",array[cur_idx]);
                int q=quantize_api(&orig,pred,abseb);
                array[cur_idx]=orig;
                
               
                final_qs[q_count++]=q;
                if (q==0){
                    final_us[u_count++]=orig;
                }
                /*
                if (cur_idx==0||cur_idx==262657||cur_idx==263657||cur_idx==264657||cur_idx==265657||cur_idx==300000||cur_idx==400000){
                    printf("curidx: %d\n",cur_idx);
                    printf("q: %d\n",q);
                    printf("pred: %f\n",orig);
                    
                     }
                */
                   
                    
                

            }
        }
    }

    
   
    //printf("%d\n",q_count);
    
    writeIntData_inBytes(final_qs,nbEle,strcat(inputpath,".q"),&status);
    writeFloatData_inBytes(final_us,u_count,strcat(inputpath,".u"),&status);
    writeFloatData_inBytes(array,nbEle,strcat(inputpath,".d"),&status);
    free(final_qs);
    free(final_us);
    free(array);
    free(ckpts);
   

    return 0;
}