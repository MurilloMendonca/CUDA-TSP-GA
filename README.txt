Nas configurações atuais do código. ele carrega uma matriz de distâncias para um caxeiro viajante assimétrico. A matriz se encontra
no arquivo mat.dat em forma de arquivo binário gerado pelo código:


void CriaMatrizRandom(int t)
{
float *v =  new float[t*t];
int r=0;
for(int i=0;i<t*t;i++)
{
    if(i%t ==r)
    {
        v[i] =0;
        r++;
        if(i+1<t*t)
        {
            v[i+1] = 1;
            i++;
        }
    }
    else
        v[i] = num_aleatorio(1,1000);
}

Onde garantimos que a diagonal principal seja formadas por zeros e temos uma diagonal ao lado da lateral feita de uns. Essa condição
foi adotada de forma a formarmos um melhor caso artificial, apenas para termos certeza que o GA vai encontrar o melhor caminho.

Para execução deve-se compilar o arquivo main.cu da forma: nvcc -o [nomeDesejadoDoExecutavel] main.cu

Demais informações disponíveis em https://docs.google.com/presentation/d/1iOtA7maJ6-LegWTzeUFu7THN7wWA88oQCT4xTskj6rA/edit?usp=sharing
