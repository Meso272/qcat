#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ByteToolkit.h"
#include "rw.h"
#include "qcat.h"
#include "zstd.h"
#include <sz_dummy_compression.h>



int main(int argc, char * argv[])
{
    int status = 0;
    //char dataType[4];
    char FilePath[640];

    if(argc < 2)
    {
        printf("Test case: huffmanZstd  [data file] [number of elements] [optional:quantBinCapacity]\n");
        
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
    
    sprintf(FilePath, "%s", argv[1]);


    size_t nbEle=atoi(argv[2]);
    //printf("%d\n",nbEle);
    int quantBinCapacity=32768;

    if(argc>=4)
        {quantBinCapacity = atoi(argv[3]);}

    int level=3;
    if(argc>=5)
        level=atoi(argv[4]);
   
    int *data = readInt32Data(FilePath, &nbEle, &status);
    //for(int i=0;i<nbEle;i++)
    //    {printf("%d\n",data[i]);}
    size_t compressedsize;
    if (quantBinCapacity>0)
        compressedsize= shortHuffmanAndZstd(0, data, quantBinCapacity, nbEle);
    else{
        unsigned char *source=(unsigned char*)malloc(nbEle);
        unsigned char *dest=(unsigned char*)malloc(nbEle);
        for (int i=0;i<nbEle;i++)
            source[i]=data[i];
        compressedsize=ZSTD_compress(dest, nbEle, source, nbEle, level);
        free(source);
        free(dest);

       }


    float cr=(float)(nbEle)*sizeof(int)/compressedsize;
    printf("%f\n",cr);
    return 0;
    

}


   
    
