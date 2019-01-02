#include <stdio.h>
 
int main(int argc, char **argv)
{
  int a,b;
  int x;
 
  printf("Enter two numbers\n");
  scanf("%d %d", &a, &b);
 
  if (a > b)
    x = 1;
  else
    x = 2;

  return x;
}
