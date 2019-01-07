FILENAME?=tritype
CC?=gcc
override CFLAGS+=-fno-optimize-sibling-calls -fno-strict-aliasing -fno-asm
override LDFLAGS+=-lm -s
all:
	$(CC) $(CFLAGS) -c $(FILENAME).c -o $(FILENAME).o
	$(CC) $(FILENAME).o -o $(FILENAME) $(LDFLAGS)
.PHONY: clean
clean:
	rm -f *.o $(FILENAME)
