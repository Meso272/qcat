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

float lorenzo_1d(float * array, int x_start, float error_bound, 
    int x_size, int block_size, int * qs, float * us, int *u_count, float * preds)//eb is absolute
{
    int x_end=x_start+block_size;
    if (x_end>x_size) x_end=x_size;
    
    float loss=0;
    int qidx=0,uidx=0;
    for (int x=x_start;x<x_end;x++){
        
        int idx=x;
        int idx_depadded=x-1;
        float orig=array[idx];
        float a=array[idx-1];
        
        float pred=a;
        preds[idx_depadded]=pred;
        loss+=fabs(orig-pred);
        int q=quantize_api(&orig,pred,error_bound);
        qs[qidx++]=q;
        if (q==0)
            us[uidx++]=orig;
        array[idx]=orig;

    (*u_count)=uidx;
    return loss;
    }
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

int main(int argc,char*argv[]){
    if(argc < 2)
    {
        printf("Test case: compress [origpath][reconpath][vrrel error_bound][blocksize][dim][dim1][dim2][opt dim3][opt:compress mod 0=all 1=nn 2=lorenzo]\n");
        
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


    char origpath[640],reconpath[640];
    sprintf(origpath, "%s", argv[1]);
    sprintf(reconpath, "%s", argv[2]);
    float eb=atof(argv[3]);
    size_t block_size=atoi(argv[4]);
    int dim=atoi(argv[5]);
    int dimx=atoi(argv[6]);
    int dimy=atoi(argv[7]);
    int dimz=0;
    if (dim==3) dimz=atoi(argv[8]);
    int compressmode=0;
    if (dim==2&&argc>=9) 
        compressmode=atoi(argv[8]);
    if (dim==3&&argc>=10) 
        compressmode=atoi(argv[9]);

    size_t nbEle,nbEle_padded;
    int status=0;

    float * origdata=readFloatData(origpath,&nbEle_padded,&status);
    
    float * recondata=readFloatData(reconpath,&nbEle,&status);
    float * decompdata=(float *)malloc(nbEle*sizeof(float));
    float * predictdata=(float *)malloc(nbEle*sizeof(float));
    printf("%d %d\n",nbEle,nbEle_padded);
    float min=2e30;
    float max=-2e30;
    if (dim==2){
        for(int x=1;x<=dimx;x++){
            for (int y=1;y<=dimy;y++){
                int idx=x*(dimy+1)+y;
                float ele=origdata[idx];
                max=ele>max?ele:max;
                min=ele<min?ele:min;


            }
        }
    }
    else{
        for(int x=1;x<=dimx;x++){
            for (int y=1;y<=dimy;y++){
                for(int z=1;z<=dimz;z++){
                    int idx=x*(dimy+1)*(dimz+1)+y*(dimz+1)+z;
                    float ele=origdata[idx];
                    max=ele>max?ele:max;
                    min=ele<min?ele:min;
                }


            }
        }

    }
    float range=max-min;
    
    float abseb=eb*range;
    printf("%f\n",abseb);
    int lorenzo_count=0;
    int nn_count=0;
    int q_count=0,u_count=0;
    int * final_qs=(int*)malloc(nbEle*sizeof(int));
    float * final_us=(float*)malloc(nbEle*sizeof(float));
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    if (dim==2){

        for(int x=1;x<=dimx;x+=block_size){
            for(int y=1;y<=dimy;y+=block_size){
                int q_block[block_size*block_size];
                float u_block[block_size*block_size];
                float origblock[block_size*block_size];
                int ublock_count=0;
                int x_end=x+block_size;
                if (x_end>dimx+1) x_end=dimx+1;
                int y_end=y+block_size;
                if (y_end>dimy+1) y_end=dimy+1;
                int cur_bs=(x_end-x)*(y_end-y);
                float blockmin=2e30,blockmax=-2e30,blocksum=0;
                float loss1=0;
                //printf("blockmeanand loss1\n");
                for (int i=x;i<x_end;i++){
                    for(int j=y;j<y_end;j++){
                        int ori_idx=i*(dimy+1)+j,recon_idx=(i-1)*dimy+j-1,block_idx=(i-x)*block_size+(j-y);
                        float ori_ele=origdata[ori_idx],recon_ele=recondata[recon_idx];
                        origblock[block_idx]=ori_ele;
                        blockmin=ori_ele<blockmin?ori_ele:blockmin;
                        blockmax=ori_ele>blockmax?ori_ele:blockmax;
                        loss1+=fabs(ori_ele-recon_ele);
                        blocksum+=ori_ele;

                    }
                }
                
                if (blockmax-blockmin<=abseb&&compressmode!=1){
                    //printf("mean\n");
                    //printf("%d %d %d %d\n",x,y,x_end,y_end);
                    lorenzo_count++;
                    float mean=blocksum/cur_bs;
                    for(int i=0;i<cur_bs;i++)
                        final_qs[q_count++]=radius;
                        //printf("%d\n",q_count);
                    for (int i=x;i<x_end;i++){
                        for(int j=y;j<y_end;j++){
                            //printf("%d %d\n",i*(dimy+1)+j,(i-1)*dimy+j-1);
                            origdata[i*(dimy+1)+j]=mean;
                            decompdata[(i-1)*dimy+j-1]=mean;
                            predictdata[(i-1)*dimy+j-1]=mean;
                        }
                    }

                    continue;
                }
                
                //printf("loss2\n");
                float loss2;
                if(compressmode!=1)
                    loss2=lorenzo_2d(origdata, x,y,abseb, dimx+1, dimy+1, block_size, q_block, u_block, &ublock_count,predictdata);
                else
                    loss2=loss1+1;
                //printf("%f %f\n",loss1,loss2);
                if (loss2<=loss1 || compressmode==2){
                //if(0){
                    //printf("lorenzo\n");
                    lorenzo_count++;
                    for(int i=0;i<cur_bs;i++)
                        final_qs[q_count++]=q_block[i];
                    for(int i=0;i<ublock_count;i++)
                        final_us[u_count++]=u_block[i];
                    for (int i=x;i<x_end;i++){
                        for(int j=y;j<y_end;j++){
                            decompdata[(i-1)*dimy+j-1]=origdata[i*(dimy+1)+j];
                        }
                    }
                }
                else{
                    //printf("nn\n");
                    nn_count++;
                    for (int i=x;i<x_end;i++){
                        for(int j=y;j<y_end;j++){
                            int ori_idx=i*(dimy+1)+j,recon_idx=(i-1)*dimy+j-1,block_idx=(i-x)*block_size+(j-y);
                            float ori_ele=origblock[block_idx],recon_ele=recondata[recon_idx];

                            int q=quantize_api(&ori_ele,recon_ele,abseb);

                            origdata[ori_idx]=ori_ele;
                            decompdata[recon_idx]=ori_ele;
                            predictdata[recon_idx]=recon_ele;
                            final_qs[q_count++]=q;
                            if (q==0)
                                final_us[u_count++]=ori_ele;
                            

                        }
                    }


                }
                


            }
        }


    }
    else{
        int x_stride_pad=(dimy+1)*(dimz+1),x_stride=dimy*dimz,y_stride_pad=dimz+1,y_stride=dimz;
        for(int x=1;x<=dimx;x+=block_size){
            for(int y=1;y<=dimy;y+=block_size){
                for(int z=1;z<=dimz;z+=block_size){
                    int q_block[block_size*block_size*block_size];
                    float u_block[block_size*block_size*block_size];
                    float origblock[block_size*block_size*block_size];
                    int ublock_count=0;
                    int x_end=x+block_size;
                    if (x_end>dimx+1) x_end=dimx+1;
                    int y_end=y+block_size;
                    if (y_end>dimy+1) y_end=dimy+1;
                    int z_end=z+block_size;
                    if (z_end>dimz+1) z_end=dimz+1;
                    int cur_bs=(x_end-x)*(y_end-y)*(z_end-z);
                    float blockmin=2e30,blockmax=-2e30,blocksum=0;
                    float loss1=0;
                    for (int i=x;i<x_end;i++){
                        for(int j=y;j<y_end;j++){
                            for(int k=z;k<z_end;k++){
                                int ori_idx=i*x_stride_pad+j*y_stride_pad+k,recon_idx=(i-1)*x_stride+(j-1)*y_stride+(k-1),block_idx=(i-x)*block_size*block_size+(j-y)*block_size+(k-z);
                                float ori_ele=origdata[ori_idx],recon_ele=recondata[recon_idx];
                                origblock[block_idx]=ori_ele;
                                blockmin=ori_ele<blockmin?ori_ele:blockmin;
                                blockmax=ori_ele>blockmax?ori_ele:blockmax;
                                loss1+=fabs(ori_ele-recon_ele);
                                blocksum+=ori_ele;
                            }

                        }
                    }
                    
                    if (blockmax-blockmin<=abseb && compressmode!=1){
                        lorenzo_count++;
                        float mean=blocksum/cur_bs;
                        //printf("%f\n",mean);
                        for(int i=0;i<cur_bs;i++)
                            final_qs[q_count++]=radius;
                        for(int i=x;i<x_end;i++){
                            for(int j=y;j<y_end;j++){
                                for(int k=z;k<z_end;k++){

                                    origdata[i*x_stride_pad+j*y_stride_pad+k]=mean;
                                    decompdata[(i-1)*x_stride+(j-1)*y_stride+(k-1)]=mean;
                                    predictdata[(i-1)*x_stride+(j-1)*y_stride+(k-1)]=mean;
                                }

                            }
                        }

                        continue;
                        
                    }
                    
                    float loss2;
                    if(compressmode!=1)
                        loss2=lorenzo_3d(origdata, x,y, z,abseb, dimx+1, dimy+1, dimz+1,block_size, q_block, u_block, &ublock_count,predictdata);
                    else loss2=loss1+1;
                    if (loss2<=loss1 || compressmode==2){
                        lorenzo_count++;
                        for(int i=0;i<cur_bs;i++)
                            final_qs[q_count++]=q_block[i];
                        for(int i=0;i<ublock_count;i++)
                            final_us[u_count++]=u_block[i];
                        for (int i=x;i<x_end;i++){
                            for(int j=y;j<y_end;j++){
                                for(int k=z;k<z_end;k++){
                                    decompdata[(i-1)*x_stride+(j-1)*y_stride+(k-1)]=origdata[i*x_stride_pad+j*y_stride_pad+k];

                                }

                            }
                        }
                    }
                    else{
                        nn_count++;
                        for (int i=x;i<x_end;i++){
                            for(int j=y;j<y_end;j++){
                                for(int k=z;k<z_end;k++){
                                    int ori_idx=i*x_stride_pad+j*y_stride_pad+k,recon_idx=(i-1)*x_stride+(j-1)*y_stride+(k-1),block_idx=(i-x)*block_size*block_size+(j-y)*block_size+(k-z);
                                    float ori_ele=origblock[block_idx],recon_ele=recondata[recon_idx];
    
                                    int q=quantize_api(&ori_ele,recon_ele,abseb);
    
                                    origdata[ori_idx]=ori_ele;
                                    decompdata[recon_idx]=ori_ele;
                                    predictdata[recon_idx]=recon_ele;
                                    final_qs[q_count++]=q;
                                    if (q==0)
                                        final_us[u_count++]=ori_ele;
                                }
                            

                            }
                        }


                    }
                

                }
            }
        }

    }  
    gettimeofday(&end,NULL); 
    double time=end.tv_sec-start.tv_sec+(end.tv_usec-start.tv_usec)/(double)1000000;

    //printf("%d\n",q_count);
    printf("%d blocks used NN.\n",nn_count);
    printf("%d blocks used Lorenzo.\n",lorenzo_count);
    printf("%.4f\n",time);
    writeIntData_inBytes(final_qs,nbEle,strcat(origpath,".q"),&status);
    writeFloatData_inBytes(final_us,u_count,strcat(origpath,".u"),&status);
    writeFloatData_inBytes(decompdata,nbEle,strcat(origpath,".d"),&status);
    writeFloatData_inBytes(predictdata,nbEle,"predicted_value.dat",&status);
    free(final_qs);
    free(final_us);
    free(decompdata);
    free(origdata);
    free(recondata);
    free(predictdata);


    return 0;
}