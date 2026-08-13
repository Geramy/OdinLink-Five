# Thunderbolt hardware — what to buy

OdinLink needs a real **Intel Thunderbolt NHI** (USB4 class `0c0340` in
`lspci`). A USB-C port or a generic USB4 card is not enough.

Run this on the Linux box first:

```bash
scripts/tb-hw-check.sh
```

## Cards are brand-locked

Thunderbolt add-in cards talk to a **vendor-specific motherboard
header**. Mixing brands does not work.

| Card | Works on |
|------|----------|
| ASUS ThunderboltEX 5 | ASUS boards on ASUS’s list (often Intel 800-series) |
| Gigabyte THUNDERBOLTS 5 | Gigabyte boards with their TB headers |
| MSI ThunderboltM4 (TB4, ~40 Gb/s) | MSI Intel 500/600/700 with **JTBT + USB 2.0** |
| MSI ThunderboltM5 (TB5) | **Not sold separately.** Bundled with MEG Z890 GODLIKE only |

Do **not** buy an ASUS or Gigabyte TB5 card for an MSI board.

## Known machines in this project

| Machine | TB today | What to do |
|---------|----------|------------|
| Minisforum **MS-S1 MAX** | Onboard Intel TB5 (`8086:5781`) | Use it. No add-in card. |
| Apple Silicon **MacBook** (TB5) | Built-in Apple NHI | Other end of the cable. See [`PYTORCH.md`](PYTORCH.md). |
| MSI **MPG Z690 Carbon WiFi** (MS-7D30) | No NHI | **ThunderboltM4** if JTBT1 is free. No retail TB5 card. |

Z690 Carbon WiFi has a **JTBT1** header on the bottom edge (near RGB).
Confirm it is unused, plus a free **JUSB** and a PCIe x4+ slot. The
13600KF has no iGPU — DP passthrough to the card is only needed if you
want monitors on the TB ports, not for OdinLink DMA.

## USB4-only cards

ASMedia ASM4242 and similar “USB4” AICs usually have **no Intel NHI**.
Linux will not create a Thunderbolt domain. OdinLink will not bind.
Skip them.

## Speeds (honest)

| Link | Typical OdinLink / bridge |
|------|---------------------------|
| TB5 (two Linux boxes, OdinLink READY) | ~40 Gb/s DMA (field: ~43 Gb/s) |
| TB4 AIC (ThunderboltM4) | ~40 Gb/s class, often less |
| TB5 IP to a Mac (`bridge/`) | ~25–40 Gb/s copies, not DMA |

For Ubuntu + PyTorch using **Mac RAM**, you do not need a new card if
the Linux trainer already has TB4/TB5. You need a TB5-rated cable and
Thunderbolt Bridge on the Mac. See [`PYTORCH.md`](PYTORCH.md).
