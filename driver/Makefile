KDIR := /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

install:
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install
	depmod -a

insmod: all
	sudo insmod odl_tb5.ko

rmmod:
	sudo rmmod odl_tb5

.PHONY: all clean install insmod rmmod
