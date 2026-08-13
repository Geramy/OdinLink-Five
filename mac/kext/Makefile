# OdinLinkRDMA.kext — build on macOS with Xcode command-line tools.
# SIP must be off to load an unsigned kext.
#
#   make
#   sudo make load
#   sudo make unload

KEXTNAME := OdinLinkRDMA
BUNDLE   := $(KEXTNAME).kext
IDENT    := com.odinlink.rdma

SDK      ?= macosx
ARCHS    ?= arm64
MINOS    ?= 14.0

CC       := xcrun -sdk $(SDK) clang
CXX      := xcrun -sdk $(SDK) clang++

CPPFLAGS := -DKERNEL -DKERNEL_PRIVATE -DDRIVER_PRIVATE \
	    -I. -I../include
CXXFLAGS := -arch $(ARCHS) -mmacosx-version-min=$(MINOS) \
	    -std=gnu++17 -fno-rtti -fno-exceptions -fno-builtin \
	    -mkernel -nostdinc \
	    -isystem $(shell xcrun -sdk $(SDK) --show-sdk-path)/System/Library/Frameworks/Kernel.framework/Headers \
	    -Wall -Wextra -Wno-unused-parameter
LDFLAGS  := -arch $(ARCHS) -mmacosx-version-min=$(MINOS) \
	    -mkernel -nostdlib -lkmodc++ -lkmod -lcc_kext

OBJS := OdinLinkRDMA.o OdinLinkNHI.o

.PHONY: all clean load unload

all: $(BUNDLE)

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(KEXTNAME): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS)

$(BUNDLE): $(KEXTNAME) Info.plist
	rm -rf $(BUNDLE)
	mkdir -p $(BUNDLE)/Contents/MacOS
	cp $(KEXTNAME) $(BUNDLE)/Contents/MacOS/
	cp Info.plist $(BUNDLE)/Contents/
	@echo "built $(BUNDLE)"

load: $(BUNDLE)
	sudo cp -R $(BUNDLE) /tmp/
	sudo kextutil -v /tmp/$(BUNDLE)

unload:
	sudo kextunload -b $(IDENT) || true

clean:
	rm -rf $(OBJS) $(KEXTNAME) $(BUNDLE)
