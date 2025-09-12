#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Sinh mảng SNR dB
void SNR_dB_range(int start,int stop,int step,int arr[]){
    int index =0 ;
    for(int i = start ; i< stop;i += step){
        arr[index] =  i; //Tạo mảng SNR dB chạy từ start đến stop với step
        index++;
    }
}
//Sinh bit ngẫu nhiên
void generate_bits(int *bits, int count){
    for(int i = 0 ; i< count;i++){
        bits[i] = rand()%2; // Tạo ra 2 bit 
    }
}
// Ánh xạ bit thành ký hiệu BPSK
void map_bpsk(int *bits, double *symbols,int count){
    for(int i = 0 ; i<count ;i++){
        if(bits[i]==0){
            symbols[i] = -1.0 ; // Ánh xạ bit 0 thành -1
        }
        else{
            symbols[i] = 1.0 ; // Ánh xạ bit 1 thành 1
        }
    }
}
// Hàm tạo số ngẫu nhiên phân phối chuẩn (Gaussian)
double guassian_random(){
    double u1 = ((double)rand()+ 1.0)/((double)RAND_MAX + 1.0);
    double u2 = ((double)rand()+ 1.0)/((double)RAND_MAX + 1.0);
    return sqrt(-2.0*log(u1))*cos(2.0*M_PI*u2); // Trả về số ngẫu nhiên phân phối chuẩn
}
// Tạo nhiễu phức
void generate_noise(double complex *noise,int count ,double N0){
    for(int i =0;i<count ;i++){
        double real = guassian_random()*sqrt(N0/2.0);
        double imag = guassian_random()*sqrt(N0/2.0);
        noise[i] = real + I*imag; // Tạo ra nhiễu phức với I là đơn vị ảo
    }
}
// Tạo kênh Rayleigh
void generate_rayleigh_channel(double complex *h,int count){
    for(int i =0; i <count ;i++){
        double real = guassian_random()/sqrt(2.0);
        double imag = guassian_random()/sqrt(2.0);
        h[i] = real + I*imag; // Tạo ra kênh Rayleigh với I là đơn vị ảo
    }
}
// Hàm truyền tín hiệu qua kênh Rayleigh với nhiễu
void transmit_signal(double *symbols,double complex *h,double complex *noise,double complex *received,int count){
    for(int i = 0 ; i<count;i++){
        received[i] = h[i]*symbols[i] + noise[i] ; // Tín hiệu nhận được là tích của kênh và ký hiệu cộng với nhiễu
    }
}
// Hàm cân bằng kênh
void equalize_channel(double complex *received,double complex *h,double complex *equalized,int count){
    for(int i=0;i<count;i++){
        equalized[i] = received[i]/h[i]; // Cân bằng kênh bằng cách chia tín hiệu nhận được cho kênh
    }
}

// Hàm giải mã BPSK
void demodulate_bpsk(double complex *equalized,int *decoded_bits,int count){
    for(int i=0;i<count;i++){
        if(creal(equalized[i]) >0){
            decoded_bits[i] = 1; // Giải mã thành bit 1 nếu phần thực lớn hơn 0
        }
        else{ decoded_bits[i]=0;} // Ngược lại giải mã thành bit 0
    }
}
//Tinh toán BER
double calculate_ber(int *original_bits,int *decoded_bits,int count){
    int error_count = 0;
    for (int i = 0; i < count; i++){
        if(original_bits[i] != decoded_bits[i]){
            error_count++;
        }
    }
    return (double)error_count / count; // Trả về tỷ lệ lỗi bit
    
}
// Hàm chính
int main(){
    srand(time(NULL)); // Khoi tạo bộ sinh số ngẫu nhiên.
    int num_bits = 100000; // Số lượng bit
    int *bits = (int *)malloc(num_bits*sizeof(int)); // Mảng lưu trữ bit
    double *symbols =(double *)malloc(num_bits * sizeof(double)); // Mảng lưu trữ ký hiệu BPSK
    double complex *h = (double complex *)malloc(num_bits * sizeof(double complex)); // Mảng lưu trữ kênh Rayleigh
    double complex *noise = (double complex *)malloc(num_bits * sizeof(double complex)); // Mảng lưu trữ nhiễu
    double complex *received = (double complex *)malloc(num_bits * sizeof(double complex)); // Mảng lưu trữ tín hiệu nhận được
    double complex *equalized = (double complex *)malloc(num_bits * sizeof(double complex)); // Mảng lưu trữ tín hiệu sau khi cân bằng kênh
    int *decoded_bits = (int *)malloc(num_bits*sizeof(int)); // Mảng lưu trữ bit đã giải mã
    int SNR_dB[11]; // Mảng lưu trữ giá trị SNR dB
    SNR_dB_range(0,21,2,SNR_dB); // Tạo mảng SNR dB từ 0 đến 20 với bước nhảy 2
    //Mở file để ghi kết quả
    FILE *fp = fopen("ber_results.txt","w");
    if(fp == NULL){
        printf("Error opening file!\n");
        return 1;
    }
    fprintf(fp,"SNR(dB)\tBER\n");// Ghi tiêu đề vào file
    printf("SNR(dB)\tBER\n"); // In tiêu
    for(int i=0;i<11;i++){
        int SNR_linear = (int)pow(10.0,SNR_dB[i]/10.0); // Chuyển đổi SNR từ dB sang tỷ lệ tuyến tính
        double N0 = 1.0/SNR_linear; // Tính toán N0
        generate_bits(bits,num_bits); // Tạo bit ngẫu nhiên
        map_bpsk(bits,symbols,num_bits); // Ánh xạ bit thành ký hiệu BPSK
        generate_noise(noise,num_bits,N0); // Tạo nhiễu phức
        generate_rayleigh_channel(h,num_bits); // Tạo kênh Rayleigh
        transmit_signal(symbols,h,noise,received,num_bits);// Truyền tín hiệu qua kênh Rayleigh với nhiễu
        equalize_channel(received,h,equalized,num_bits); // Cân bằng kênh
        demodulate_bpsk(equalized,decoded_bits,num_bits); // Giải mã BPSK
        double BER = calculate_ber(bits,decoded_bits,num_bits); // Tính toán BER
        printf("%d\t%lf\n",SNR_dB[i],BER); // In ra SNR dB và BER tương ứng
        // Ghi kết quả vào file
        fprintf(fp,"%d\t%lf\n",SNR_dB[i],BER);
        printf("Simulation for SNR = %d dB completed.\n", SNR_dB[i]);
    }
    fclose(fp); // Đóng file
    // Giải phóng bộ nhớ
    free(bits);
    free(symbols);
    free(decoded_bits);
    free(h);
    free(noise);
    free(received);
    free(equalized);
return 0;
}