
evidence/vx-row-tile/probe.o:     file format elf64-x86-64


Disassembly of section .ltext:

0000000000000000 <integrated>:
       0:	55                   	push   %rbp
       1:	41 57                	push   %r15
       3:	41 56                	push   %r14
       5:	41 55                	push   %r13
       7:	41 54                	push   %r12
       9:	53                   	push   %rbx
       a:	48 83 ec 38          	sub    $0x38,%rsp
       e:	48 8d 05 f9 ff ff ff 	lea    -0x7(%rip),%rax        # e <integrated+0xe>
      15:	49 bd 00 00 00 00 00 	movabs $0x0,%r13
      1c:	00 00 00 
      1f:	bb ff ff ff ff       	mov    $0xffffffff,%ebx
      24:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
      29:	48 89 0c 24          	mov    %rcx,(%rsp)
      2d:	49 01 c5             	add    %rax,%r13
      30:	89 c8                	mov    %ecx,%eax
      32:	44 09 c0             	or     %r8d,%eax
      35:	0f 88 5a 07 00 00    	js     795 <integrated+0x795>
      3b:	48 8b 04 24          	mov    (%rsp),%rax
      3f:	41 89 c2             	mov    %eax,%r10d
      42:	41 c1 ea 03          	shr    $0x3,%r10d
      46:	0f 84 35 02 00 00    	je     281 <integrated+0x281>
      4c:	44 89 d0             	mov    %r10d,%eax
      4f:	45 85 c0             	test   %r8d,%r8d
      52:	0f 84 96 04 00 00    	je     4ee <integrated+0x4ee>
      58:	48 b9 00 00 00 00 00 	movabs $0x0,%rcx
      5f:	00 00 00 
      62:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
      67:	45 89 c1             	mov    %r8d,%r9d
      6a:	45 89 cb             	mov    %r9d,%r11d
      6d:	62 d2 7d 28 7c c0    	vpbroadcastd %r8d,%ymm0
      73:	41 83 e3 07          	and    $0x7,%r11d
      77:	41 81 e1 f8 ff ff 7f 	and    $0x7ffffff8,%r9d
      7e:	31 db                	xor    %ebx,%ebx
      80:	c4 c1 7d 6f 4c 0d 00 	vmovdqa 0x0(%r13,%rcx,1),%ymm1
      87:	eb 23                	jmp    ac <integrated+0xac>
      89:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
      90:	48 89 d9             	mov    %rbx,%rcx
      93:	48 c1 e1 23          	shl    $0x23,%rcx
      97:	48 ff c3             	inc    %rbx
      9a:	48 c1 f9 1e          	sar    $0x1e,%rcx
      9e:	c5 fc 11 1c 0f       	vmovups %ymm3,(%rdi,%rcx,1)
      a3:	48 39 c3             	cmp    %rax,%rbx
      a6:	0f 84 d5 01 00 00    	je     281 <integrated+0x281>
      ac:	8d 0c dd 00 00 00 00 	lea    0x0(,%rbx,8),%ecx
      b3:	62 f2 7d 28 7c d1    	vpbroadcastd %ecx,%ymm2
      b9:	c5 ed eb d1          	vpor   %ymm1,%ymm2,%ymm2
      bd:	c4 e2 7d 40 d2       	vpmulld %ymm2,%ymm0,%ymm2
      c2:	41 83 f8 08          	cmp    $0x8,%r8d
      c6:	73 18                	jae    e0 <integrated+0xe0>
      c8:	45 31 f6             	xor    %r14d,%r14d
      cb:	c5 e0 57 db          	vxorps %xmm3,%xmm3,%xmm3
      cf:	e9 75 01 00 00       	jmp    249 <integrated+0x249>
      d4:	66 66 66 2e 0f 1f 84 	data16 data16 cs nopw 0x0(%rax,%rax,1)
      db:	00 00 00 00 00 
      e0:	c5 e0 57 db          	vxorps %xmm3,%xmm3,%xmm3
      e4:	45 31 f6             	xor    %r14d,%r14d
      e7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
      ee:	00 00 
      f0:	62 d2 7d 28 7c e6    	vpbroadcastd %r14d,%ymm4
      f6:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
      fa:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
      fe:	41 8d 4e 01          	lea    0x1(%r14),%ecx
     102:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
     106:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
     10d:	62 b1 54 38 59 24 b2 	vmulps (%rdx,%r14,4){1to8},%ymm5,%ymm4
     114:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
     118:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     11c:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
     120:	62 f2 7d 28 7c e1    	vpbroadcastd %ecx,%ymm4
     126:	41 8d 4e 02          	lea    0x2(%r14),%ecx
     12a:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
     12e:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
     135:	62 b1 54 38 59 64 b2 	vmulps 0x4(%rdx,%r14,4){1to8},%ymm5,%ymm4
     13c:	01 
     13d:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
     141:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     145:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
     149:	62 f2 7d 28 7c e1    	vpbroadcastd %ecx,%ymm4
     14f:	41 8d 4e 03          	lea    0x3(%r14),%ecx
     153:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
     157:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
     15e:	62 b1 54 38 59 64 b2 	vmulps 0x8(%rdx,%r14,4){1to8},%ymm5,%ymm4
     165:	02 
     166:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
     16a:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     16e:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
     172:	62 f2 7d 28 7c e1    	vpbroadcastd %ecx,%ymm4
     178:	41 8d 4e 04          	lea    0x4(%r14),%ecx
     17c:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
     180:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
     187:	62 b1 54 38 59 64 b2 	vmulps 0xc(%rdx,%r14,4){1to8},%ymm5,%ymm4
     18e:	03 
     18f:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
     193:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     197:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
     19b:	62 f2 7d 28 7c e1    	vpbroadcastd %ecx,%ymm4
     1a1:	41 8d 4e 05          	lea    0x5(%r14),%ecx
     1a5:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
     1a9:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
     1b0:	62 b1 54 38 59 64 b2 	vmulps 0x10(%rdx,%r14,4){1to8},%ymm5,%ymm4
     1b7:	04 
     1b8:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
     1bc:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     1c0:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
     1c4:	62 f2 7d 28 7c e1    	vpbroadcastd %ecx,%ymm4
     1ca:	41 8d 4e 06          	lea    0x6(%r14),%ecx
     1ce:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
     1d2:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
     1d9:	62 b1 54 38 59 64 b2 	vmulps 0x14(%rdx,%r14,4){1to8},%ymm5,%ymm4
     1e0:	05 
     1e1:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
     1e5:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     1e9:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
     1ed:	62 f2 7d 28 7c e1    	vpbroadcastd %ecx,%ymm4
     1f3:	41 8d 4e 07          	lea    0x7(%r14),%ecx
     1f7:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
     1fb:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
     202:	62 b1 54 38 59 64 b2 	vmulps 0x18(%rdx,%r14,4){1to8},%ymm5,%ymm4
     209:	06 
     20a:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
     20e:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     212:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
     216:	62 f2 7d 28 7c e1    	vpbroadcastd %ecx,%ymm4
     21c:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
     220:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
     227:	62 b1 54 38 59 64 b2 	vmulps 0x1c(%rdx,%r14,4){1to8},%ymm5,%ymm4
     22e:	07 
     22f:	49 83 c6 08          	add    $0x8,%r14
     233:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
     237:	4d 39 ce             	cmp    %r9,%r14
     23a:	0f 85 b0 fe ff ff    	jne    f0 <integrated+0xf0>
     240:	4d 85 db             	test   %r11,%r11
     243:	0f 84 47 fe ff ff    	je     90 <integrated+0x90>
     249:	4c 89 d9             	mov    %r11,%rcx
     24c:	0f 1f 40 00          	nopl   0x0(%rax)
     250:	62 d2 7d 28 7c e6    	vpbroadcastd %r14d,%ymm4
     256:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
     25a:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     25e:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
     262:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
     269:	62 b1 54 38 59 24 b2 	vmulps (%rdx,%r14,4){1to8},%ymm5,%ymm4
     270:	49 ff c6             	inc    %r14
     273:	48 ff c9             	dec    %rcx
     276:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
     27a:	75 d4                	jne    250 <integrated+0x250>
     27c:	e9 0f fe ff ff       	jmp    90 <integrated+0x90>
     281:	4c 8b 0c 24          	mov    (%rsp),%r9
     285:	44 89 cd             	mov    %r9d,%ebp
     288:	41 d1 e9             	shr    %r9d
     28b:	81 e5 f8 ff ff 7f    	and    $0x7ffffff8,%ebp
     291:	44 89 cb             	mov    %r9d,%ebx
     294:	83 e3 03             	and    $0x3,%ebx
     297:	0f 84 4a 02 00 00    	je     4e7 <integrated+0x4e7>
     29d:	45 85 c0             	test   %r8d,%r8d
     2a0:	0f 8e f5 02 00 00    	jle    59b <integrated+0x59b>
     2a6:	b9 03 1c 00 00       	mov    $0x1c03,%ecx
     2ab:	44 8d 65 01          	lea    0x1(%rbp),%r12d
     2af:	89 e8                	mov    %ebp,%eax
     2b1:	48 89 6c 24 18       	mov    %rbp,0x18(%rsp)
     2b6:	45 89 c3             	mov    %r8d,%r11d
     2b9:	45 89 df             	mov    %r11d,%r15d
     2bc:	41 83 e7 07          	and    $0x7,%r15d
     2c0:	41 81 e3 f8 ff ff 7f 	and    $0x7ffffff8,%r11d
     2c7:	43 8d 3c 00          	lea    (%r8,%r8,1),%edi
     2cb:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
     2d0:	89 d8                	mov    %ebx,%eax
     2d2:	48 89 5c 24 10       	mov    %rbx,0x10(%rsp)
     2d7:	4c 89 6c 24 20       	mov    %r13,0x20(%rsp)
     2dc:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
     2e1:	31 db                	xor    %ebx,%ebx
     2e3:	c4 e2 70 f7 2c 24    	bextr  %ecx,(%rsp),%ebp
     2e9:	45 0f af e0          	imul   %r8d,%r12d
     2ed:	41 0f af e8          	imul   %r8d,%ebp
     2f1:	c1 e5 03             	shl    $0x3,%ebp
     2f4:	eb 32                	jmp    328 <integrated+0x328>
     2f6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
     2fd:	00 00 00 
     300:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
     305:	4c 8b 4c 24 08       	mov    0x8(%rsp),%r9
     30a:	49 01 fc             	add    %rdi,%r12
     30d:	48 01 fd             	add    %rdi,%rbp
     310:	48 8d 0c 58          	lea    (%rax,%rbx,2),%rcx
     314:	48 ff c3             	inc    %rbx
     317:	c4 c1 78 13 04 89    	vmovlps %xmm0,(%r9,%rcx,4)
     31d:	48 3b 5c 24 28       	cmp    0x28(%rsp),%rbx
     322:	0f 84 ab 01 00 00    	je     4d3 <integrated+0x4d3>
     328:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
     32c:	45 31 f6             	xor    %r14d,%r14d
     32f:	41 83 f8 08          	cmp    $0x8,%r8d
     333:	0f 82 53 01 00 00    	jb     48c <integrated+0x48c>
     339:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
     340:	42 8d 44 35 00       	lea    0x0(%rbp,%r14,1),%eax
     345:	43 8d 0c 34          	lea    (%r12,%r14,1),%ecx
     349:	48 98                	cltq   
     34b:	48 63 c9             	movslq %ecx,%rcx
     34e:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
     353:	42 8d 44 35 01       	lea    0x1(%rbp,%r14,1),%eax
     358:	c4 e3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%rcx,4),%xmm1,%xmm1
     35f:	43 8d 4c 34 01       	lea    0x1(%r12,%r14,1),%ecx
     364:	62 b1 74 18 59 0c b2 	vmulps (%rdx,%r14,4){1to4},%xmm1,%xmm1
     36b:	48 98                	cltq   
     36d:	48 63 c9             	movslq %ecx,%rcx
     370:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
     374:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
     379:	42 8d 44 35 02       	lea    0x2(%rbp,%r14,1),%eax
     37e:	c4 e3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%rcx,4),%xmm1,%xmm1
     385:	43 8d 4c 34 02       	lea    0x2(%r12,%r14,1),%ecx
     38a:	62 b1 74 18 59 4c b2 	vmulps 0x4(%rdx,%r14,4){1to4},%xmm1,%xmm1
     391:	01 
     392:	48 98                	cltq   
     394:	48 63 c9             	movslq %ecx,%rcx
     397:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
     39b:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
     3a0:	42 8d 44 35 03       	lea    0x3(%rbp,%r14,1),%eax
     3a5:	c4 e3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%rcx,4),%xmm1,%xmm1
     3ac:	43 8d 4c 34 03       	lea    0x3(%r12,%r14,1),%ecx
     3b1:	62 b1 74 18 59 4c b2 	vmulps 0x8(%rdx,%r14,4){1to4},%xmm1,%xmm1
     3b8:	02 
     3b9:	48 98                	cltq   
     3bb:	48 63 c9             	movslq %ecx,%rcx
     3be:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
     3c2:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
     3c7:	42 8d 44 35 04       	lea    0x4(%rbp,%r14,1),%eax
     3cc:	c4 e3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%rcx,4),%xmm1,%xmm1
     3d3:	43 8d 4c 34 04       	lea    0x4(%r12,%r14,1),%ecx
     3d8:	62 b1 74 18 59 4c b2 	vmulps 0xc(%rdx,%r14,4){1to4},%xmm1,%xmm1
     3df:	03 
     3e0:	48 98                	cltq   
     3e2:	48 63 c9             	movslq %ecx,%rcx
     3e5:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
     3e9:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
     3ee:	42 8d 44 35 05       	lea    0x5(%rbp,%r14,1),%eax
     3f3:	c4 e3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%rcx,4),%xmm1,%xmm1
     3fa:	43 8d 4c 34 05       	lea    0x5(%r12,%r14,1),%ecx
     3ff:	62 b1 74 18 59 4c b2 	vmulps 0x10(%rdx,%r14,4){1to4},%xmm1,%xmm1
     406:	04 
     407:	48 98                	cltq   
     409:	48 63 c9             	movslq %ecx,%rcx
     40c:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
     410:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
     415:	42 8d 44 35 06       	lea    0x6(%rbp,%r14,1),%eax
     41a:	c4 e3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%rcx,4),%xmm1,%xmm1
     421:	43 8d 4c 34 06       	lea    0x6(%r12,%r14,1),%ecx
     426:	62 b1 74 18 59 4c b2 	vmulps 0x14(%rdx,%r14,4){1to4},%xmm1,%xmm1
     42d:	05 
     42e:	48 98                	cltq   
     430:	48 63 c9             	movslq %ecx,%rcx
     433:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
     437:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
     43c:	42 8d 44 35 07       	lea    0x7(%rbp,%r14,1),%eax
     441:	c4 e3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%rcx,4),%xmm1,%xmm1
     448:	43 8d 4c 34 07       	lea    0x7(%r12,%r14,1),%ecx
     44d:	62 b1 74 18 59 4c b2 	vmulps 0x18(%rdx,%r14,4){1to4},%xmm1,%xmm1
     454:	06 
     455:	48 98                	cltq   
     457:	48 63 c9             	movslq %ecx,%rcx
     45a:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
     45e:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
     463:	c4 e3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%rcx,4),%xmm1,%xmm1
     46a:	62 b1 74 18 59 4c b2 	vmulps 0x1c(%rdx,%r14,4){1to4},%xmm1,%xmm1
     471:	07 
     472:	49 83 c6 08          	add    $0x8,%r14
     476:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
     47a:	4d 39 f3             	cmp    %r14,%r11
     47d:	0f 85 bd fe ff ff    	jne    340 <integrated+0x340>
     483:	4d 85 ff             	test   %r15,%r15
     486:	0f 84 74 fe ff ff    	je     300 <integrated+0x300>
     48c:	47 8d 0c 34          	lea    (%r12,%r14,1),%r9d
     490:	42 8d 4c 35 00       	lea    0x0(%rbp,%r14,1),%ecx
     495:	4e 8d 34 b2          	lea    (%rdx,%r14,4),%r14
     499:	31 c0                	xor    %eax,%eax
     49b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
     4a0:	44 8d 14 01          	lea    (%rcx,%rax,1),%r10d
     4a4:	45 8d 2c 01          	lea    (%r9,%rax,1),%r13d
     4a8:	4d 63 d2             	movslq %r10d,%r10
     4ab:	4d 63 ed             	movslq %r13d,%r13
     4ae:	c4 a1 7a 10 0c 96    	vmovss (%rsi,%r10,4),%xmm1
     4b4:	c4 a3 71 21 0c ae 10 	vinsertps $0x10,(%rsi,%r13,4),%xmm1,%xmm1
     4bb:	62 d1 74 18 59 0c 86 	vmulps (%r14,%rax,4){1to4},%xmm1,%xmm1
     4c2:	48 ff c0             	inc    %rax
     4c5:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
     4c9:	49 39 c7             	cmp    %rax,%r15
     4cc:	75 d2                	jne    4a0 <integrated+0x4a0>
     4ce:	e9 2d fe ff ff       	jmp    300 <integrated+0x300>
     4d3:	4c 8b 6c 24 20       	mov    0x20(%rsp),%r13
     4d8:	48 8b 6c 24 18       	mov    0x18(%rsp),%rbp
     4dd:	48 8b 5c 24 10       	mov    0x10(%rsp),%rbx
     4e2:	e9 f7 00 00 00       	jmp    5de <integrated+0x5de>
     4e7:	31 db                	xor    %ebx,%ebx
     4e9:	e9 f0 00 00 00       	jmp    5de <integrated+0x5de>
     4ee:	41 89 c1             	mov    %eax,%r9d
     4f1:	41 83 e1 07          	and    $0x7,%r9d
     4f5:	83 3c 24 40          	cmpl   $0x40,(%rsp)
     4f9:	73 05                	jae    500 <integrated+0x500>
     4fb:	45 31 db             	xor    %r11d,%r11d
     4fe:	eb 5d                	jmp    55d <integrated+0x55d>
     500:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
     505:	25 f8 ff ff 0f       	and    $0xffffff8,%eax
     50a:	45 31 db             	xor    %r11d,%r11d
     50d:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
     511:	48 81 c1 e0 00 00 00 	add    $0xe0,%rcx
     518:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
     51f:	00 
     520:	62 f1 fe 48 7f 81 20 	vmovdqu64 %zmm0,-0xe0(%rcx)
     527:	ff ff ff 
     52a:	62 f1 fe 48 7f 81 60 	vmovdqu64 %zmm0,-0xa0(%rcx)
     531:	ff ff ff 
     534:	62 f1 fe 48 7f 81 a0 	vmovdqu64 %zmm0,-0x60(%rcx)
     53b:	ff ff ff 
     53e:	62 f1 fe 48 7f 81 e0 	vmovdqu64 %zmm0,-0x20(%rcx)
     545:	ff ff ff 
     548:	49 83 c3 08          	add    $0x8,%r11
     54c:	48 81 c1 00 01 00 00 	add    $0x100,%rcx
     553:	4c 39 d8             	cmp    %r11,%rax
     556:	75 c8                	jne    520 <integrated+0x520>
     558:	4d 85 c9             	test   %r9,%r9
     55b:	74 22                	je     57f <integrated+0x57f>
     55d:	49 c1 e3 05          	shl    $0x5,%r11
     561:	4c 03 5c 24 08       	add    0x8(%rsp),%r11
     566:	41 c1 e1 05          	shl    $0x5,%r9d
     56a:	31 c0                	xor    %eax,%eax
     56c:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
     570:	c4 c1 7e 7f 04 03    	vmovdqu %ymm0,(%r11,%rax,1)
     576:	48 83 c0 20          	add    $0x20,%rax
     57a:	49 39 c1             	cmp    %rax,%r9
     57d:	75 f1                	jne    570 <integrated+0x570>
     57f:	4c 8b 0c 24          	mov    (%rsp),%r9
     583:	44 89 cd             	mov    %r9d,%ebp
     586:	41 d1 e9             	shr    %r9d
     589:	81 e5 f8 ff ff 7f    	and    $0x7ffffff8,%ebp
     58f:	44 89 cb             	mov    %r9d,%ebx
     592:	83 e3 03             	and    $0x3,%ebx
     595:	0f 84 bc 01 00 00    	je     757 <integrated+0x757>
     59b:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
     5a0:	41 c1 e1 03          	shl    $0x3,%r9d
     5a4:	44 89 d0             	mov    %r10d,%eax
     5a7:	48 c1 e0 05          	shl    $0x5,%rax
     5ab:	49 89 f7             	mov    %rsi,%r15
     5ae:	49 89 d6             	mov    %rdx,%r14
     5b1:	4d 89 c4             	mov    %r8,%r12
     5b4:	31 f6                	xor    %esi,%esi
     5b6:	41 83 e1 18          	and    $0x18,%r9d
     5ba:	4c 89 ca             	mov    %r9,%rdx
     5bd:	48 01 c8             	add    %rcx,%rax
     5c0:	48 b9 00 00 00 00 00 	movabs $0x0,%rcx
     5c7:	00 00 00 
     5ca:	48 89 c7             	mov    %rax,%rdi
     5cd:	c5 f8 77             	vzeroupper 
     5d0:	41 ff 54 0d 00       	call   *0x0(%r13,%rcx,1)
     5d5:	4d 89 e0             	mov    %r12,%r8
     5d8:	4c 89 f2             	mov    %r14,%rdx
     5db:	4c 89 fe             	mov    %r15,%rsi
     5de:	44 8d 4c 5d 00       	lea    0x0(%rbp,%rbx,2),%r9d
     5e3:	44 3b 0c 24          	cmp    (%rsp),%r9d
     5e7:	0f 8d a6 01 00 00    	jge    793 <integrated+0x793>
     5ed:	45 85 c0             	test   %r8d,%r8d
     5f0:	0f 8e 5d 01 00 00    	jle    753 <integrated+0x753>
     5f6:	8b 0c 24             	mov    (%rsp),%ecx
     5f9:	44 89 c8             	mov    %r9d,%eax
     5fc:	45 0f af c8          	imul   %r8d,%r9d
     600:	45 89 c2             	mov    %r8d,%r10d
     603:	45 89 d3             	mov    %r10d,%r11d
     606:	44 89 d3             	mov    %r10d,%ebx
     609:	41 83 e3 07          	and    $0x7,%r11d
     60d:	81 e3 f8 ff ff 7f    	and    $0x7ffffff8,%ebx
     613:	eb 24                	jmp    639 <integrated+0x639>
     615:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
     61c:	00 00 00 00 
     620:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
     625:	4d 01 d1             	add    %r10,%r9
     628:	c5 fa 11 04 87       	vmovss %xmm0,(%rdi,%rax,4)
     62d:	48 ff c0             	inc    %rax
     630:	48 39 c8             	cmp    %rcx,%rax
     633:	0f 83 5a 01 00 00    	jae    793 <integrated+0x793>
     639:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
     63d:	45 31 f6             	xor    %r14d,%r14d
     640:	41 83 f8 08          	cmp    $0x8,%r8d
     644:	0f 82 da 00 00 00    	jb     724 <integrated+0x724>
     64a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
     650:	43 8d 3c 31          	lea    (%r9,%r14,1),%edi
     654:	48 63 ff             	movslq %edi,%rdi
     657:	c5 fa 10 0c be       	vmovss (%rsi,%rdi,4),%xmm1
     65c:	43 8d 7c 31 01       	lea    0x1(%r9,%r14,1),%edi
     661:	c4 a1 72 59 0c b2    	vmulss (%rdx,%r14,4),%xmm1,%xmm1
     667:	48 63 ff             	movslq %edi,%rdi
     66a:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     66e:	c5 fa 10 0c be       	vmovss (%rsi,%rdi,4),%xmm1
     673:	43 8d 7c 31 02       	lea    0x2(%r9,%r14,1),%edi
     678:	c4 a1 72 59 4c b2 04 	vmulss 0x4(%rdx,%r14,4),%xmm1,%xmm1
     67f:	48 63 ff             	movslq %edi,%rdi
     682:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     686:	c5 fa 10 0c be       	vmovss (%rsi,%rdi,4),%xmm1
     68b:	43 8d 7c 31 03       	lea    0x3(%r9,%r14,1),%edi
     690:	c4 a1 72 59 4c b2 08 	vmulss 0x8(%rdx,%r14,4),%xmm1,%xmm1
     697:	48 63 ff             	movslq %edi,%rdi
     69a:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     69e:	c5 fa 10 0c be       	vmovss (%rsi,%rdi,4),%xmm1
     6a3:	43 8d 7c 31 04       	lea    0x4(%r9,%r14,1),%edi
     6a8:	c4 a1 72 59 4c b2 0c 	vmulss 0xc(%rdx,%r14,4),%xmm1,%xmm1
     6af:	48 63 ff             	movslq %edi,%rdi
     6b2:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     6b6:	c5 fa 10 0c be       	vmovss (%rsi,%rdi,4),%xmm1
     6bb:	43 8d 7c 31 05       	lea    0x5(%r9,%r14,1),%edi
     6c0:	c4 a1 72 59 4c b2 10 	vmulss 0x10(%rdx,%r14,4),%xmm1,%xmm1
     6c7:	48 63 ff             	movslq %edi,%rdi
     6ca:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     6ce:	c5 fa 10 0c be       	vmovss (%rsi,%rdi,4),%xmm1
     6d3:	43 8d 7c 31 06       	lea    0x6(%r9,%r14,1),%edi
     6d8:	c4 a1 72 59 4c b2 14 	vmulss 0x14(%rdx,%r14,4),%xmm1,%xmm1
     6df:	48 63 ff             	movslq %edi,%rdi
     6e2:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     6e6:	c5 fa 10 0c be       	vmovss (%rsi,%rdi,4),%xmm1
     6eb:	43 8d 7c 31 07       	lea    0x7(%r9,%r14,1),%edi
     6f0:	c4 a1 72 59 4c b2 18 	vmulss 0x18(%rdx,%r14,4),%xmm1,%xmm1
     6f7:	48 63 ff             	movslq %edi,%rdi
     6fa:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     6fe:	c5 fa 10 0c be       	vmovss (%rsi,%rdi,4),%xmm1
     703:	c4 a1 72 59 4c b2 1c 	vmulss 0x1c(%rdx,%r14,4),%xmm1,%xmm1
     70a:	49 83 c6 08          	add    $0x8,%r14
     70e:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     712:	4c 39 f3             	cmp    %r14,%rbx
     715:	0f 85 35 ff ff ff    	jne    650 <integrated+0x650>
     71b:	4d 85 db             	test   %r11,%r11
     71e:	0f 84 fc fe ff ff    	je     620 <integrated+0x620>
     724:	4e 8d 3c b2          	lea    (%rdx,%r14,4),%r15
     728:	45 01 ce             	add    %r9d,%r14d
     72b:	45 31 e4             	xor    %r12d,%r12d
     72e:	66 90                	xchg   %ax,%ax
     730:	43 8d 3c 26          	lea    (%r14,%r12,1),%edi
     734:	48 63 ff             	movslq %edi,%rdi
     737:	c5 fa 10 0c be       	vmovss (%rsi,%rdi,4),%xmm1
     73c:	c4 81 72 59 0c a7    	vmulss (%r15,%r12,4),%xmm1,%xmm1
     742:	49 ff c4             	inc    %r12
     745:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     749:	4d 39 e3             	cmp    %r12,%r11
     74c:	75 e2                	jne    730 <integrated+0x730>
     74e:	e9 cd fe ff ff       	jmp    620 <integrated+0x620>
     753:	01 db                	add    %ebx,%ebx
     755:	eb 0a                	jmp    761 <integrated+0x761>
     757:	31 db                	xor    %ebx,%ebx
     759:	41 89 e9             	mov    %ebp,%r9d
     75c:	3b 2c 24             	cmp    (%rsp),%ebp
     75f:	74 34                	je     795 <integrated+0x795>
     761:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
     766:	44 89 c9             	mov    %r9d,%ecx
     769:	f7 d3                	not    %ebx
     76b:	31 f6                	xor    %esi,%esi
     76d:	48 8d 3c 88          	lea    (%rax,%rcx,4),%rdi
     771:	48 8b 04 24          	mov    (%rsp),%rax
     775:	01 d8                	add    %ebx,%eax
     777:	29 e8                	sub    %ebp,%eax
     779:	48 8d 14 85 04 00 00 	lea    0x4(,%rax,4),%rdx
     780:	00 
     781:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
     788:	00 00 00 
     78b:	c5 f8 77             	vzeroupper 
     78e:	41 ff 54 05 00       	call   *0x0(%r13,%rax,1)
     793:	31 db                	xor    %ebx,%ebx
     795:	89 d8                	mov    %ebx,%eax
     797:	48 83 c4 38          	add    $0x38,%rsp
     79b:	5b                   	pop    %rbx
     79c:	41 5c                	pop    %r12
     79e:	41 5d                	pop    %r13
     7a0:	41 5e                	pop    %r14
     7a2:	41 5f                	pop    %r15
     7a4:	5d                   	pop    %rbp
     7a5:	c5 f8 77             	vzeroupper 
     7a8:	c3                   	ret    
     7a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000007b0 <tile8>:
     7b0:	48 8d 05 f9 ff ff ff 	lea    -0x7(%rip),%rax        # 7b0 <tile8>
     7b7:	49 b9 00 00 00 00 00 	movabs $0x0,%r9
     7be:	00 00 00 
     7c1:	41 89 ca             	mov    %ecx,%r10d
     7c4:	49 01 c1             	add    %rax,%r9
     7c7:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
     7cc:	45 09 c2             	or     %r8d,%r10d
     7cf:	0f 88 b3 04 00 00    	js     c88 <tile8+0x4d8>
     7d5:	55                   	push   %rbp
     7d6:	41 57                	push   %r15
     7d8:	41 56                	push   %r14
     7da:	41 55                	push   %r13
     7dc:	41 54                	push   %r12
     7de:	53                   	push   %rbx
     7df:	50                   	push   %rax
     7e0:	89 c8                	mov    %ecx,%eax
     7e2:	c1 e8 03             	shr    $0x3,%eax
     7e5:	0f 84 36 02 00 00    	je     a21 <tile8+0x271>
     7eb:	41 89 c2             	mov    %eax,%r10d
     7ee:	45 85 c0             	test   %r8d,%r8d
     7f1:	0f 84 bd 03 00 00    	je     bb4 <tile8+0x404>
     7f7:	49 bf 00 00 00 00 00 	movabs $0x0,%r15
     7fe:	00 00 00 
     801:	45 89 c3             	mov    %r8d,%r11d
     804:	44 89 db             	mov    %r11d,%ebx
     807:	62 d2 7d 28 7c c0    	vpbroadcastd %r8d,%ymm0
     80d:	83 e3 07             	and    $0x7,%ebx
     810:	41 81 e3 f8 ff ff 7f 	and    $0x7ffffff8,%r11d
     817:	45 31 f6             	xor    %r14d,%r14d
     81a:	c4 81 7d 6f 0c 39    	vmovdqa (%r9,%r15,1),%ymm1
     820:	eb 2b                	jmp    84d <tile8+0x9d>
     822:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
     829:	1f 84 00 00 00 00 00 
     830:	4d 89 f7             	mov    %r14,%r15
     833:	49 c1 e7 23          	shl    $0x23,%r15
     837:	49 ff c6             	inc    %r14
     83a:	49 c1 ff 1e          	sar    $0x1e,%r15
     83e:	c4 a1 7c 11 1c 3f    	vmovups %ymm3,(%rdi,%r15,1)
     844:	4d 39 d6             	cmp    %r10,%r14
     847:	0f 84 d4 01 00 00    	je     a21 <tile8+0x271>
     84d:	42 8d 2c f5 00 00 00 	lea    0x0(,%r14,8),%ebp
     854:	00 
     855:	62 f2 7d 28 7c d5    	vpbroadcastd %ebp,%ymm2
     85b:	c5 ed eb d1          	vpor   %ymm1,%ymm2,%ymm2
     85f:	c4 e2 7d 40 d2       	vpmulld %ymm2,%ymm0,%ymm2
     864:	41 83 f8 08          	cmp    $0x8,%r8d
     868:	73 16                	jae    880 <tile8+0xd0>
     86a:	45 31 ff             	xor    %r15d,%r15d
     86d:	c5 e0 57 db          	vxorps %xmm3,%xmm3,%xmm3
     871:	e9 73 01 00 00       	jmp    9e9 <tile8+0x239>
     876:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
     87d:	00 00 00 
     880:	c5 e0 57 db          	vxorps %xmm3,%xmm3,%xmm3
     884:	45 31 ff             	xor    %r15d,%r15d
     887:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
     88e:	00 00 
     890:	62 d2 7d 28 7c e7    	vpbroadcastd %r15d,%ymm4
     896:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
     89a:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     89e:	41 8d 6f 01          	lea    0x1(%r15),%ebp
     8a2:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
     8a6:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
     8ad:	62 b1 54 38 59 24 ba 	vmulps (%rdx,%r15,4){1to8},%ymm5,%ymm4
     8b4:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
     8b8:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     8bc:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
     8c0:	62 f2 7d 28 7c e5    	vpbroadcastd %ebp,%ymm4
     8c6:	41 8d 6f 02          	lea    0x2(%r15),%ebp
     8ca:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
     8ce:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
     8d5:	62 b1 54 38 59 64 ba 	vmulps 0x4(%rdx,%r15,4){1to8},%ymm5,%ymm4
     8dc:	01 
     8dd:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
     8e1:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     8e5:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
     8e9:	62 f2 7d 28 7c e5    	vpbroadcastd %ebp,%ymm4
     8ef:	41 8d 6f 03          	lea    0x3(%r15),%ebp
     8f3:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
     8f7:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
     8fe:	62 b1 54 38 59 64 ba 	vmulps 0x8(%rdx,%r15,4){1to8},%ymm5,%ymm4
     905:	02 
     906:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
     90a:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     90e:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
     912:	62 f2 7d 28 7c e5    	vpbroadcastd %ebp,%ymm4
     918:	41 8d 6f 04          	lea    0x4(%r15),%ebp
     91c:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
     920:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
     927:	62 b1 54 38 59 64 ba 	vmulps 0xc(%rdx,%r15,4){1to8},%ymm5,%ymm4
     92e:	03 
     92f:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
     933:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     937:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
     93b:	62 f2 7d 28 7c e5    	vpbroadcastd %ebp,%ymm4
     941:	41 8d 6f 05          	lea    0x5(%r15),%ebp
     945:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
     949:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
     950:	62 b1 54 38 59 64 ba 	vmulps 0x10(%rdx,%r15,4){1to8},%ymm5,%ymm4
     957:	04 
     958:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
     95c:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     960:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
     964:	62 f2 7d 28 7c e5    	vpbroadcastd %ebp,%ymm4
     96a:	41 8d 6f 06          	lea    0x6(%r15),%ebp
     96e:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
     972:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
     979:	62 b1 54 38 59 64 ba 	vmulps 0x14(%rdx,%r15,4){1to8},%ymm5,%ymm4
     980:	05 
     981:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
     985:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     989:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
     98d:	62 f2 7d 28 7c e5    	vpbroadcastd %ebp,%ymm4
     993:	41 8d 6f 07          	lea    0x7(%r15),%ebp
     997:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
     99b:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
     9a2:	62 b1 54 38 59 64 ba 	vmulps 0x18(%rdx,%r15,4){1to8},%ymm5,%ymm4
     9a9:	06 
     9aa:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
     9ae:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     9b2:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
     9b6:	62 f2 7d 28 7c e5    	vpbroadcastd %ebp,%ymm4
     9bc:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
     9c0:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
     9c7:	62 b1 54 38 59 64 ba 	vmulps 0x1c(%rdx,%r15,4){1to8},%ymm5,%ymm4
     9ce:	07 
     9cf:	49 83 c7 08          	add    $0x8,%r15
     9d3:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
     9d7:	4d 39 df             	cmp    %r11,%r15
     9da:	0f 85 b0 fe ff ff    	jne    890 <tile8+0xe0>
     9e0:	48 85 db             	test   %rbx,%rbx
     9e3:	0f 84 47 fe ff ff    	je     830 <tile8+0x80>
     9e9:	49 89 dc             	mov    %rbx,%r12
     9ec:	0f 1f 40 00          	nopl   0x0(%rax)
     9f0:	62 d2 7d 28 7c e7    	vpbroadcastd %r15d,%ymm4
     9f6:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
     9fa:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     9fe:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
     a02:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
     a09:	62 b1 54 38 59 24 ba 	vmulps (%rdx,%r15,4){1to8},%ymm5,%ymm4
     a10:	49 ff c7             	inc    %r15
     a13:	49 ff cc             	dec    %r12
     a16:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
     a1a:	75 d4                	jne    9f0 <tile8+0x240>
     a1c:	e9 0f fe ff ff       	jmp    830 <tile8+0x80>
     a21:	41 89 ca             	mov    %ecx,%r10d
     a24:	41 81 e2 f8 ff ff 7f 	and    $0x7ffffff8,%r10d
     a2b:	41 39 ca             	cmp    %ecx,%r10d
     a2e:	0f 8d 44 02 00 00    	jge    c78 <tile8+0x4c8>
     a34:	45 85 c0             	test   %r8d,%r8d
     a37:	0f 8e 11 02 00 00    	jle    c4e <tile8+0x49e>
     a3d:	bd 03 1c 00 00       	mov    $0x1c03,%ebp
     a42:	89 c8                	mov    %ecx,%eax
     a44:	41 89 c9             	mov    %ecx,%r9d
     a47:	45 89 c2             	mov    %r8d,%r10d
     a4a:	45 89 d3             	mov    %r10d,%r11d
     a4d:	44 89 d3             	mov    %r10d,%ebx
     a50:	25 f8 ff ff 7f       	and    $0x7ffffff8,%eax
     a55:	41 83 e3 07          	and    $0x7,%r11d
     a59:	81 e3 f8 ff ff 7f    	and    $0x7ffffff8,%ebx
     a5f:	c4 e2 50 f7 c9       	bextr  %ebp,%ecx,%ecx
     a64:	41 0f af c8          	imul   %r8d,%ecx
     a68:	c1 e1 03             	shl    $0x3,%ecx
     a6b:	eb 17                	jmp    a84 <tile8+0x2d4>
     a6d:	0f 1f 00             	nopl   (%rax)
     a70:	c5 fa 11 04 87       	vmovss %xmm0,(%rdi,%rax,4)
     a75:	48 ff c0             	inc    %rax
     a78:	4c 01 d1             	add    %r10,%rcx
     a7b:	4c 39 c8             	cmp    %r9,%rax
     a7e:	0f 84 f4 01 00 00    	je     c78 <tile8+0x4c8>
     a84:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
     a88:	45 31 f6             	xor    %r14d,%r14d
     a8b:	41 83 f8 08          	cmp    $0x8,%r8d
     a8f:	0f 82 e7 00 00 00    	jb     b7c <tile8+0x3cc>
     a95:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
     a9c:	00 00 00 00 
     aa0:	42 8d 2c 31          	lea    (%rcx,%r14,1),%ebp
     aa4:	4c 63 fd             	movslq %ebp,%r15
     aa7:	42 8d 6c 31 01       	lea    0x1(%rcx,%r14,1),%ebp
     aac:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
     ab2:	c4 a1 72 59 0c b2    	vmulss (%rdx,%r14,4),%xmm1,%xmm1
     ab8:	4c 63 fd             	movslq %ebp,%r15
     abb:	42 8d 6c 31 02       	lea    0x2(%rcx,%r14,1),%ebp
     ac0:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     ac4:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
     aca:	4c 63 fd             	movslq %ebp,%r15
     acd:	42 8d 6c 31 03       	lea    0x3(%rcx,%r14,1),%ebp
     ad2:	c4 a1 72 59 4c b2 04 	vmulss 0x4(%rdx,%r14,4),%xmm1,%xmm1
     ad9:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     add:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
     ae3:	4c 63 fd             	movslq %ebp,%r15
     ae6:	42 8d 6c 31 04       	lea    0x4(%rcx,%r14,1),%ebp
     aeb:	c4 a1 72 59 4c b2 08 	vmulss 0x8(%rdx,%r14,4),%xmm1,%xmm1
     af2:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     af6:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
     afc:	4c 63 fd             	movslq %ebp,%r15
     aff:	42 8d 6c 31 05       	lea    0x5(%rcx,%r14,1),%ebp
     b04:	c4 a1 72 59 4c b2 0c 	vmulss 0xc(%rdx,%r14,4),%xmm1,%xmm1
     b0b:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     b0f:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
     b15:	4c 63 fd             	movslq %ebp,%r15
     b18:	42 8d 6c 31 06       	lea    0x6(%rcx,%r14,1),%ebp
     b1d:	c4 a1 72 59 4c b2 10 	vmulss 0x10(%rdx,%r14,4),%xmm1,%xmm1
     b24:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     b28:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
     b2e:	4c 63 fd             	movslq %ebp,%r15
     b31:	42 8d 6c 31 07       	lea    0x7(%rcx,%r14,1),%ebp
     b36:	c4 a1 72 59 4c b2 14 	vmulss 0x14(%rdx,%r14,4),%xmm1,%xmm1
     b3d:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     b41:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
     b47:	4c 63 fd             	movslq %ebp,%r15
     b4a:	c4 a1 72 59 4c b2 18 	vmulss 0x18(%rdx,%r14,4),%xmm1,%xmm1
     b51:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     b55:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
     b5b:	c4 a1 72 59 4c b2 1c 	vmulss 0x1c(%rdx,%r14,4),%xmm1,%xmm1
     b62:	49 83 c6 08          	add    $0x8,%r14
     b66:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     b6a:	4c 39 f3             	cmp    %r14,%rbx
     b6d:	0f 85 2d ff ff ff    	jne    aa0 <tile8+0x2f0>
     b73:	4d 85 db             	test   %r11,%r11
     b76:	0f 84 f4 fe ff ff    	je     a70 <tile8+0x2c0>
     b7c:	4e 8d 3c b2          	lea    (%rdx,%r14,4),%r15
     b80:	41 01 ce             	add    %ecx,%r14d
     b83:	45 31 e4             	xor    %r12d,%r12d
     b86:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
     b8d:	00 00 00 
     b90:	43 8d 2c 26          	lea    (%r14,%r12,1),%ebp
     b94:	4c 63 ed             	movslq %ebp,%r13
     b97:	c4 a1 7a 10 0c ae    	vmovss (%rsi,%r13,4),%xmm1
     b9d:	c4 81 72 59 0c a7    	vmulss (%r15,%r12,4),%xmm1,%xmm1
     ba3:	49 ff c4             	inc    %r12
     ba6:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     baa:	4d 39 e3             	cmp    %r12,%r11
     bad:	75 e1                	jne    b90 <tile8+0x3e0>
     baf:	e9 bc fe ff ff       	jmp    a70 <tile8+0x2c0>
     bb4:	44 89 d2             	mov    %r10d,%edx
     bb7:	83 e2 07             	and    $0x7,%edx
     bba:	83 f9 40             	cmp    $0x40,%ecx
     bbd:	73 04                	jae    bc3 <tile8+0x413>
     bbf:	31 f6                	xor    %esi,%esi
     bc1:	eb 5a                	jmp    c1d <tile8+0x46d>
     bc3:	41 81 e2 f8 ff ff 0f 	and    $0xffffff8,%r10d
     bca:	4c 8d 87 e0 00 00 00 	lea    0xe0(%rdi),%r8
     bd1:	31 f6                	xor    %esi,%esi
     bd3:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
     bd7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
     bde:	00 00 
     be0:	62 d1 fe 48 7f 80 20 	vmovdqu64 %zmm0,-0xe0(%r8)
     be7:	ff ff ff 
     bea:	62 d1 fe 48 7f 80 60 	vmovdqu64 %zmm0,-0xa0(%r8)
     bf1:	ff ff ff 
     bf4:	62 d1 fe 48 7f 80 a0 	vmovdqu64 %zmm0,-0x60(%r8)
     bfb:	ff ff ff 
     bfe:	62 d1 fe 48 7f 80 e0 	vmovdqu64 %zmm0,-0x20(%r8)
     c05:	ff ff ff 
     c08:	48 83 c6 08          	add    $0x8,%rsi
     c0c:	49 81 c0 00 01 00 00 	add    $0x100,%r8
     c13:	49 39 f2             	cmp    %rsi,%r10
     c16:	75 c8                	jne    be0 <tile8+0x430>
     c18:	48 85 d2             	test   %rdx,%rdx
     c1b:	74 22                	je     c3f <tile8+0x48f>
     c1d:	48 c1 e6 05          	shl    $0x5,%rsi
     c21:	c1 e2 05             	shl    $0x5,%edx
     c24:	45 31 c0             	xor    %r8d,%r8d
     c27:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
     c2b:	48 01 fe             	add    %rdi,%rsi
     c2e:	66 90                	xchg   %ax,%ax
     c30:	c4 a1 7e 7f 04 06    	vmovdqu %ymm0,(%rsi,%r8,1)
     c36:	49 83 c0 20          	add    $0x20,%r8
     c3a:	4c 39 c2             	cmp    %r8,%rdx
     c3d:	75 f1                	jne    c30 <tile8+0x480>
     c3f:	41 89 ca             	mov    %ecx,%r10d
     c42:	41 81 e2 f8 ff ff 7f 	and    $0x7ffffff8,%r10d
     c49:	41 39 ca             	cmp    %ecx,%r10d
     c4c:	74 2a                	je     c78 <tile8+0x4c8>
     c4e:	41 f7 d2             	not    %r10d
     c51:	89 c0                	mov    %eax,%eax
     c53:	48 c1 e0 05          	shl    $0x5,%rax
     c57:	31 f6                	xor    %esi,%esi
     c59:	44 01 d1             	add    %r10d,%ecx
     c5c:	48 01 c7             	add    %rax,%rdi
     c5f:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
     c66:	00 00 00 
     c69:	48 8d 14 8d 04 00 00 	lea    0x4(,%rcx,4),%rdx
     c70:	00 
     c71:	c5 f8 77             	vzeroupper 
     c74:	41 ff 14 01          	call   *(%r9,%rax,1)
     c78:	31 c0                	xor    %eax,%eax
     c7a:	48 83 c4 08          	add    $0x8,%rsp
     c7e:	5b                   	pop    %rbx
     c7f:	41 5c                	pop    %r12
     c81:	41 5d                	pop    %r13
     c83:	41 5e                	pop    %r14
     c85:	41 5f                	pop    %r15
     c87:	5d                   	pop    %rbp
     c88:	c5 f8 77             	vzeroupper 
     c8b:	c3                   	ret    
     c8c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000000c90 <tile4>:
     c90:	48 8d 05 f9 ff ff ff 	lea    -0x7(%rip),%rax        # c90 <tile4>
     c97:	49 b9 00 00 00 00 00 	movabs $0x0,%r9
     c9e:	00 00 00 
     ca1:	41 89 ca             	mov    %ecx,%r10d
     ca4:	49 01 c1             	add    %rax,%r9
     ca7:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
     cac:	45 09 c2             	or     %r8d,%r10d
     caf:	0f 88 a3 04 00 00    	js     1158 <tile4+0x4c8>
     cb5:	55                   	push   %rbp
     cb6:	41 57                	push   %r15
     cb8:	41 56                	push   %r14
     cba:	41 55                	push   %r13
     cbc:	41 54                	push   %r12
     cbe:	53                   	push   %rbx
     cbf:	50                   	push   %rax
     cc0:	89 c8                	mov    %ecx,%eax
     cc2:	c1 e8 02             	shr    $0x2,%eax
     cc5:	0f 84 36 02 00 00    	je     f01 <tile4+0x271>
     ccb:	41 89 c2             	mov    %eax,%r10d
     cce:	45 85 c0             	test   %r8d,%r8d
     cd1:	0f 84 bd 03 00 00    	je     1094 <tile4+0x404>
     cd7:	49 bf 00 00 00 00 00 	movabs $0x0,%r15
     cde:	00 00 00 
     ce1:	45 89 c3             	mov    %r8d,%r11d
     ce4:	44 89 db             	mov    %r11d,%ebx
     ce7:	62 d2 7d 08 7c c0    	vpbroadcastd %r8d,%xmm0
     ced:	83 e3 07             	and    $0x7,%ebx
     cf0:	41 81 e3 f8 ff ff 7f 	and    $0x7ffffff8,%r11d
     cf7:	45 31 f6             	xor    %r14d,%r14d
     cfa:	c4 81 79 6f 0c 39    	vmovdqa (%r9,%r15,1),%xmm1
     d00:	eb 2b                	jmp    d2d <tile4+0x9d>
     d02:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
     d09:	1f 84 00 00 00 00 00 
     d10:	4d 89 f7             	mov    %r14,%r15
     d13:	49 c1 e7 22          	shl    $0x22,%r15
     d17:	49 ff c6             	inc    %r14
     d1a:	49 c1 ff 1e          	sar    $0x1e,%r15
     d1e:	c4 a1 78 11 1c 3f    	vmovups %xmm3,(%rdi,%r15,1)
     d24:	4d 39 d6             	cmp    %r10,%r14
     d27:	0f 84 d4 01 00 00    	je     f01 <tile4+0x271>
     d2d:	42 8d 2c b5 00 00 00 	lea    0x0(,%r14,4),%ebp
     d34:	00 
     d35:	62 f2 7d 08 7c d5    	vpbroadcastd %ebp,%xmm2
     d3b:	c5 e9 eb d1          	vpor   %xmm1,%xmm2,%xmm2
     d3f:	c4 e2 79 40 d2       	vpmulld %xmm2,%xmm0,%xmm2
     d44:	41 83 f8 08          	cmp    $0x8,%r8d
     d48:	73 16                	jae    d60 <tile4+0xd0>
     d4a:	45 31 ff             	xor    %r15d,%r15d
     d4d:	c5 e0 57 db          	vxorps %xmm3,%xmm3,%xmm3
     d51:	e9 73 01 00 00       	jmp    ec9 <tile4+0x239>
     d56:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
     d5d:	00 00 00 
     d60:	c5 e0 57 db          	vxorps %xmm3,%xmm3,%xmm3
     d64:	45 31 ff             	xor    %r15d,%r15d
     d67:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
     d6e:	00 00 
     d70:	62 d2 7d 08 7c e7    	vpbroadcastd %r15d,%xmm4
     d76:	c5 fc 46 c8          	kxnorw %k0,%k0,%k1
     d7a:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     d7e:	41 8d 6f 01          	lea    0x1(%r15),%ebp
     d82:	c5 e9 fe e4          	vpaddd %xmm4,%xmm2,%xmm4
     d86:	62 f2 7d 09 92 2c a6 	vgatherdps (%rsi,%xmm4,4),%xmm5{%k1}
     d8d:	62 b1 54 18 59 24 ba 	vmulps (%rdx,%r15,4){1to4},%xmm5,%xmm4
     d94:	c5 fc 46 c8          	kxnorw %k0,%k0,%k1
     d98:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     d9c:	c5 e0 58 dc          	vaddps %xmm4,%xmm3,%xmm3
     da0:	62 f2 7d 08 7c e5    	vpbroadcastd %ebp,%xmm4
     da6:	41 8d 6f 02          	lea    0x2(%r15),%ebp
     daa:	c5 e9 fe e4          	vpaddd %xmm4,%xmm2,%xmm4
     dae:	62 f2 7d 09 92 2c a6 	vgatherdps (%rsi,%xmm4,4),%xmm5{%k1}
     db5:	62 b1 54 18 59 64 ba 	vmulps 0x4(%rdx,%r15,4){1to4},%xmm5,%xmm4
     dbc:	01 
     dbd:	c5 fc 46 c8          	kxnorw %k0,%k0,%k1
     dc1:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     dc5:	c5 e0 58 dc          	vaddps %xmm4,%xmm3,%xmm3
     dc9:	62 f2 7d 08 7c e5    	vpbroadcastd %ebp,%xmm4
     dcf:	41 8d 6f 03          	lea    0x3(%r15),%ebp
     dd3:	c5 e9 fe e4          	vpaddd %xmm4,%xmm2,%xmm4
     dd7:	62 f2 7d 09 92 2c a6 	vgatherdps (%rsi,%xmm4,4),%xmm5{%k1}
     dde:	62 b1 54 18 59 64 ba 	vmulps 0x8(%rdx,%r15,4){1to4},%xmm5,%xmm4
     de5:	02 
     de6:	c5 fc 46 c8          	kxnorw %k0,%k0,%k1
     dea:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     dee:	c5 e0 58 dc          	vaddps %xmm4,%xmm3,%xmm3
     df2:	62 f2 7d 08 7c e5    	vpbroadcastd %ebp,%xmm4
     df8:	41 8d 6f 04          	lea    0x4(%r15),%ebp
     dfc:	c5 e9 fe e4          	vpaddd %xmm4,%xmm2,%xmm4
     e00:	62 f2 7d 09 92 2c a6 	vgatherdps (%rsi,%xmm4,4),%xmm5{%k1}
     e07:	62 b1 54 18 59 64 ba 	vmulps 0xc(%rdx,%r15,4){1to4},%xmm5,%xmm4
     e0e:	03 
     e0f:	c5 fc 46 c8          	kxnorw %k0,%k0,%k1
     e13:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     e17:	c5 e0 58 dc          	vaddps %xmm4,%xmm3,%xmm3
     e1b:	62 f2 7d 08 7c e5    	vpbroadcastd %ebp,%xmm4
     e21:	41 8d 6f 05          	lea    0x5(%r15),%ebp
     e25:	c5 e9 fe e4          	vpaddd %xmm4,%xmm2,%xmm4
     e29:	62 f2 7d 09 92 2c a6 	vgatherdps (%rsi,%xmm4,4),%xmm5{%k1}
     e30:	62 b1 54 18 59 64 ba 	vmulps 0x10(%rdx,%r15,4){1to4},%xmm5,%xmm4
     e37:	04 
     e38:	c5 fc 46 c8          	kxnorw %k0,%k0,%k1
     e3c:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     e40:	c5 e0 58 dc          	vaddps %xmm4,%xmm3,%xmm3
     e44:	62 f2 7d 08 7c e5    	vpbroadcastd %ebp,%xmm4
     e4a:	41 8d 6f 06          	lea    0x6(%r15),%ebp
     e4e:	c5 e9 fe e4          	vpaddd %xmm4,%xmm2,%xmm4
     e52:	62 f2 7d 09 92 2c a6 	vgatherdps (%rsi,%xmm4,4),%xmm5{%k1}
     e59:	62 b1 54 18 59 64 ba 	vmulps 0x14(%rdx,%r15,4){1to4},%xmm5,%xmm4
     e60:	05 
     e61:	c5 fc 46 c8          	kxnorw %k0,%k0,%k1
     e65:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     e69:	c5 e0 58 dc          	vaddps %xmm4,%xmm3,%xmm3
     e6d:	62 f2 7d 08 7c e5    	vpbroadcastd %ebp,%xmm4
     e73:	41 8d 6f 07          	lea    0x7(%r15),%ebp
     e77:	c5 e9 fe e4          	vpaddd %xmm4,%xmm2,%xmm4
     e7b:	62 f2 7d 09 92 2c a6 	vgatherdps (%rsi,%xmm4,4),%xmm5{%k1}
     e82:	62 b1 54 18 59 64 ba 	vmulps 0x18(%rdx,%r15,4){1to4},%xmm5,%xmm4
     e89:	06 
     e8a:	c5 fc 46 c8          	kxnorw %k0,%k0,%k1
     e8e:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     e92:	c5 e0 58 dc          	vaddps %xmm4,%xmm3,%xmm3
     e96:	62 f2 7d 08 7c e5    	vpbroadcastd %ebp,%xmm4
     e9c:	c5 e9 fe e4          	vpaddd %xmm4,%xmm2,%xmm4
     ea0:	62 f2 7d 09 92 2c a6 	vgatherdps (%rsi,%xmm4,4),%xmm5{%k1}
     ea7:	62 b1 54 18 59 64 ba 	vmulps 0x1c(%rdx,%r15,4){1to4},%xmm5,%xmm4
     eae:	07 
     eaf:	49 83 c7 08          	add    $0x8,%r15
     eb3:	c5 e0 58 dc          	vaddps %xmm4,%xmm3,%xmm3
     eb7:	4d 39 df             	cmp    %r11,%r15
     eba:	0f 85 b0 fe ff ff    	jne    d70 <tile4+0xe0>
     ec0:	48 85 db             	test   %rbx,%rbx
     ec3:	0f 84 47 fe ff ff    	je     d10 <tile4+0x80>
     ec9:	49 89 dc             	mov    %rbx,%r12
     ecc:	0f 1f 40 00          	nopl   0x0(%rax)
     ed0:	62 d2 7d 08 7c e7    	vpbroadcastd %r15d,%xmm4
     ed6:	c5 fc 46 c8          	kxnorw %k0,%k0,%k1
     eda:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
     ede:	c5 e9 fe e4          	vpaddd %xmm4,%xmm2,%xmm4
     ee2:	62 f2 7d 09 92 2c a6 	vgatherdps (%rsi,%xmm4,4),%xmm5{%k1}
     ee9:	62 b1 54 18 59 24 ba 	vmulps (%rdx,%r15,4){1to4},%xmm5,%xmm4
     ef0:	49 ff c7             	inc    %r15
     ef3:	49 ff cc             	dec    %r12
     ef6:	c5 e0 58 dc          	vaddps %xmm4,%xmm3,%xmm3
     efa:	75 d4                	jne    ed0 <tile4+0x240>
     efc:	e9 0f fe ff ff       	jmp    d10 <tile4+0x80>
     f01:	41 89 ca             	mov    %ecx,%r10d
     f04:	41 81 e2 fc ff ff 7f 	and    $0x7ffffffc,%r10d
     f0b:	41 39 ca             	cmp    %ecx,%r10d
     f0e:	0f 8d 34 02 00 00    	jge    1148 <tile4+0x4b8>
     f14:	45 85 c0             	test   %r8d,%r8d
     f17:	0f 8e 01 02 00 00    	jle    111e <tile4+0x48e>
     f1d:	bd 02 1d 00 00       	mov    $0x1d02,%ebp
     f22:	89 c8                	mov    %ecx,%eax
     f24:	41 89 c9             	mov    %ecx,%r9d
     f27:	45 89 c2             	mov    %r8d,%r10d
     f2a:	45 89 d3             	mov    %r10d,%r11d
     f2d:	44 89 d3             	mov    %r10d,%ebx
     f30:	25 fc ff ff 7f       	and    $0x7ffffffc,%eax
     f35:	41 83 e3 07          	and    $0x7,%r11d
     f39:	81 e3 f8 ff ff 7f    	and    $0x7ffffff8,%ebx
     f3f:	c4 e2 50 f7 c9       	bextr  %ebp,%ecx,%ecx
     f44:	41 0f af c8          	imul   %r8d,%ecx
     f48:	c1 e1 02             	shl    $0x2,%ecx
     f4b:	eb 17                	jmp    f64 <tile4+0x2d4>
     f4d:	0f 1f 00             	nopl   (%rax)
     f50:	c5 fa 11 04 87       	vmovss %xmm0,(%rdi,%rax,4)
     f55:	48 ff c0             	inc    %rax
     f58:	4c 01 d1             	add    %r10,%rcx
     f5b:	4c 39 c8             	cmp    %r9,%rax
     f5e:	0f 84 e4 01 00 00    	je     1148 <tile4+0x4b8>
     f64:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
     f68:	45 31 f6             	xor    %r14d,%r14d
     f6b:	41 83 f8 08          	cmp    $0x8,%r8d
     f6f:	0f 82 e7 00 00 00    	jb     105c <tile4+0x3cc>
     f75:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
     f7c:	00 00 00 00 
     f80:	42 8d 2c 31          	lea    (%rcx,%r14,1),%ebp
     f84:	4c 63 fd             	movslq %ebp,%r15
     f87:	42 8d 6c 31 01       	lea    0x1(%rcx,%r14,1),%ebp
     f8c:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
     f92:	c4 a1 72 59 0c b2    	vmulss (%rdx,%r14,4),%xmm1,%xmm1
     f98:	4c 63 fd             	movslq %ebp,%r15
     f9b:	42 8d 6c 31 02       	lea    0x2(%rcx,%r14,1),%ebp
     fa0:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     fa4:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
     faa:	4c 63 fd             	movslq %ebp,%r15
     fad:	42 8d 6c 31 03       	lea    0x3(%rcx,%r14,1),%ebp
     fb2:	c4 a1 72 59 4c b2 04 	vmulss 0x4(%rdx,%r14,4),%xmm1,%xmm1
     fb9:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     fbd:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
     fc3:	4c 63 fd             	movslq %ebp,%r15
     fc6:	42 8d 6c 31 04       	lea    0x4(%rcx,%r14,1),%ebp
     fcb:	c4 a1 72 59 4c b2 08 	vmulss 0x8(%rdx,%r14,4),%xmm1,%xmm1
     fd2:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     fd6:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
     fdc:	4c 63 fd             	movslq %ebp,%r15
     fdf:	42 8d 6c 31 05       	lea    0x5(%rcx,%r14,1),%ebp
     fe4:	c4 a1 72 59 4c b2 0c 	vmulss 0xc(%rdx,%r14,4),%xmm1,%xmm1
     feb:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
     fef:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
     ff5:	4c 63 fd             	movslq %ebp,%r15
     ff8:	42 8d 6c 31 06       	lea    0x6(%rcx,%r14,1),%ebp
     ffd:	c4 a1 72 59 4c b2 10 	vmulss 0x10(%rdx,%r14,4),%xmm1,%xmm1
    1004:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    1008:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
    100e:	4c 63 fd             	movslq %ebp,%r15
    1011:	42 8d 6c 31 07       	lea    0x7(%rcx,%r14,1),%ebp
    1016:	c4 a1 72 59 4c b2 14 	vmulss 0x14(%rdx,%r14,4),%xmm1,%xmm1
    101d:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    1021:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
    1027:	4c 63 fd             	movslq %ebp,%r15
    102a:	c4 a1 72 59 4c b2 18 	vmulss 0x18(%rdx,%r14,4),%xmm1,%xmm1
    1031:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    1035:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
    103b:	c4 a1 72 59 4c b2 1c 	vmulss 0x1c(%rdx,%r14,4),%xmm1,%xmm1
    1042:	49 83 c6 08          	add    $0x8,%r14
    1046:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    104a:	4c 39 f3             	cmp    %r14,%rbx
    104d:	0f 85 2d ff ff ff    	jne    f80 <tile4+0x2f0>
    1053:	4d 85 db             	test   %r11,%r11
    1056:	0f 84 f4 fe ff ff    	je     f50 <tile4+0x2c0>
    105c:	4e 8d 3c b2          	lea    (%rdx,%r14,4),%r15
    1060:	41 01 ce             	add    %ecx,%r14d
    1063:	45 31 e4             	xor    %r12d,%r12d
    1066:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    106d:	00 00 00 
    1070:	43 8d 2c 26          	lea    (%r14,%r12,1),%ebp
    1074:	4c 63 ed             	movslq %ebp,%r13
    1077:	c4 a1 7a 10 0c ae    	vmovss (%rsi,%r13,4),%xmm1
    107d:	c4 81 72 59 0c a7    	vmulss (%r15,%r12,4),%xmm1,%xmm1
    1083:	49 ff c4             	inc    %r12
    1086:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    108a:	4d 39 e3             	cmp    %r12,%r11
    108d:	75 e1                	jne    1070 <tile4+0x3e0>
    108f:	e9 bc fe ff ff       	jmp    f50 <tile4+0x2c0>
    1094:	44 89 d2             	mov    %r10d,%edx
    1097:	83 e2 07             	and    $0x7,%edx
    109a:	83 f9 20             	cmp    $0x20,%ecx
    109d:	73 04                	jae    10a3 <tile4+0x413>
    109f:	31 f6                	xor    %esi,%esi
    10a1:	eb 43                	jmp    10e6 <tile4+0x456>
    10a3:	41 81 e2 f8 ff ff 1f 	and    $0x1ffffff8,%r10d
    10aa:	4c 8d 47 70          	lea    0x70(%rdi),%r8
    10ae:	31 f6                	xor    %esi,%esi
    10b0:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    10b4:	66 66 66 2e 0f 1f 84 	data16 data16 cs nopw 0x0(%rax,%rax,1)
    10bb:	00 00 00 00 00 
    10c0:	62 d1 fe 48 7f 80 90 	vmovdqu64 %zmm0,-0x70(%r8)
    10c7:	ff ff ff 
    10ca:	62 d1 fe 48 7f 80 d0 	vmovdqu64 %zmm0,-0x30(%r8)
    10d1:	ff ff ff 
    10d4:	48 83 c6 08          	add    $0x8,%rsi
    10d8:	49 83 e8 80          	sub    $0xffffffffffffff80,%r8
    10dc:	49 39 f2             	cmp    %rsi,%r10
    10df:	75 df                	jne    10c0 <tile4+0x430>
    10e1:	48 85 d2             	test   %rdx,%rdx
    10e4:	74 29                	je     110f <tile4+0x47f>
    10e6:	48 c1 e6 04          	shl    $0x4,%rsi
    10ea:	c1 e2 04             	shl    $0x4,%edx
    10ed:	45 31 c0             	xor    %r8d,%r8d
    10f0:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    10f4:	48 01 fe             	add    %rdi,%rsi
    10f7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    10fe:	00 00 
    1100:	c4 a1 7a 7f 04 06    	vmovdqu %xmm0,(%rsi,%r8,1)
    1106:	49 83 c0 10          	add    $0x10,%r8
    110a:	4c 39 c2             	cmp    %r8,%rdx
    110d:	75 f1                	jne    1100 <tile4+0x470>
    110f:	41 89 ca             	mov    %ecx,%r10d
    1112:	41 81 e2 fc ff ff 7f 	and    $0x7ffffffc,%r10d
    1119:	41 39 ca             	cmp    %ecx,%r10d
    111c:	74 2a                	je     1148 <tile4+0x4b8>
    111e:	41 f7 d2             	not    %r10d
    1121:	89 c0                	mov    %eax,%eax
    1123:	48 c1 e0 04          	shl    $0x4,%rax
    1127:	31 f6                	xor    %esi,%esi
    1129:	44 01 d1             	add    %r10d,%ecx
    112c:	48 01 c7             	add    %rax,%rdi
    112f:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    1136:	00 00 00 
    1139:	48 8d 14 8d 04 00 00 	lea    0x4(,%rcx,4),%rdx
    1140:	00 
    1141:	c5 f8 77             	vzeroupper 
    1144:	41 ff 14 01          	call   *(%r9,%rax,1)
    1148:	31 c0                	xor    %eax,%eax
    114a:	48 83 c4 08          	add    $0x8,%rsp
    114e:	5b                   	pop    %rbx
    114f:	41 5c                	pop    %r12
    1151:	41 5d                	pop    %r13
    1153:	41 5e                	pop    %r14
    1155:	41 5f                	pop    %r15
    1157:	5d                   	pop    %rbp
    1158:	c5 f8 77             	vzeroupper 
    115b:	c3                   	ret    
    115c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000001160 <tile2>:
    1160:	55                   	push   %rbp
    1161:	41 57                	push   %r15
    1163:	41 56                	push   %r14
    1165:	41 55                	push   %r13
    1167:	41 54                	push   %r12
    1169:	53                   	push   %rbx
    116a:	48 83 ec 28          	sub    $0x28,%rsp
    116e:	48 8d 05 f9 ff ff ff 	lea    -0x7(%rip),%rax        # 116e <tile2+0xe>
    1175:	49 b9 00 00 00 00 00 	movabs $0x0,%r9
    117c:	00 00 00 
    117f:	44 89 c5             	mov    %r8d,%ebp
    1182:	41 89 c8             	mov    %ecx,%r8d
    1185:	49 01 c1             	add    %rax,%r9
    1188:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    118d:	41 09 e8             	or     %ebp,%r8d
    1190:	0f 88 dd 04 00 00    	js     1673 <tile2+0x513>
    1196:	89 c8                	mov    %ecx,%eax
    1198:	d1 e8                	shr    %eax
    119a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
    119f:	4c 89 4c 24 18       	mov    %r9,0x18(%rsp)
    11a4:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
    11a9:	89 44 24 04          	mov    %eax,0x4(%rsp)
    11ad:	0f 84 20 02 00 00    	je     13d3 <tile2+0x273>
    11b3:	8b 4c 24 04          	mov    0x4(%rsp),%ecx
    11b7:	85 ed                	test   %ebp,%ebp
    11b9:	0f 84 b5 03 00 00    	je     1574 <tile2+0x414>
    11bf:	41 89 eb             	mov    %ebp,%r11d
    11c2:	44 89 db             	mov    %r11d,%ebx
    11c5:	45 89 de             	mov    %r11d,%r14d
    11c8:	83 e3 07             	and    $0x7,%ebx
    11cb:	41 81 e6 f8 ff ff 7f 	and    $0x7ffffff8,%r14d
    11d2:	8d 7c 2d 00          	lea    0x0(%rbp,%rbp,1),%edi
    11d6:	45 31 e4             	xor    %r12d,%r12d
    11d9:	45 31 ed             	xor    %r13d,%r13d
    11dc:	48 89 4c 24 20       	mov    %rcx,0x20(%rsp)
    11e1:	eb 39                	jmp    121c <tile2+0xbc>
    11e3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
    11ea:	84 00 00 00 00 00 
    11f0:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    11f5:	4c 89 e0             	mov    %r12,%rax
    11f8:	48 c1 e0 21          	shl    $0x21,%rax
    11fc:	49 ff c4             	inc    %r12
    11ff:	49 01 fb             	add    %rdi,%r11
    1202:	49 01 fd             	add    %rdi,%r13
    1205:	4c 89 c5             	mov    %r8,%rbp
    1208:	48 c1 f8 1e          	sar    $0x1e,%rax
    120c:	c5 f8 13 04 01       	vmovlps %xmm0,(%rcx,%rax,1)
    1211:	4c 3b 64 24 20       	cmp    0x20(%rsp),%r12
    1216:	0f 84 b7 01 00 00    	je     13d3 <tile2+0x273>
    121c:	49 89 e8             	mov    %rbp,%r8
    121f:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
    1223:	83 fd 08             	cmp    $0x8,%ebp
    1226:	73 08                	jae    1230 <tile2+0xd0>
    1228:	31 ed                	xor    %ebp,%ebp
    122a:	e9 5d 01 00 00       	jmp    138c <tile2+0x22c>
    122f:	90                   	nop
    1230:	31 ed                	xor    %ebp,%ebp
    1232:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
    1239:	1f 84 00 00 00 00 00 
    1240:	41 8d 44 2d 00       	lea    0x0(%r13,%rbp,1),%eax
    1245:	45 8d 0c 2b          	lea    (%r11,%rbp,1),%r9d
    1249:	48 98                	cltq   
    124b:	4d 63 c9             	movslq %r9d,%r9
    124e:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
    1253:	41 8d 44 2d 01       	lea    0x1(%r13,%rbp,1),%eax
    1258:	c4 a3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%r9,4),%xmm1,%xmm1
    125f:	45 8d 4c 2b 01       	lea    0x1(%r11,%rbp,1),%r9d
    1264:	62 f1 74 18 59 0c aa 	vmulps (%rdx,%rbp,4){1to4},%xmm1,%xmm1
    126b:	48 98                	cltq   
    126d:	4d 63 c9             	movslq %r9d,%r9
    1270:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
    1274:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
    1279:	41 8d 44 2d 02       	lea    0x2(%r13,%rbp,1),%eax
    127e:	c4 a3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%r9,4),%xmm1,%xmm1
    1285:	45 8d 4c 2b 02       	lea    0x2(%r11,%rbp,1),%r9d
    128a:	62 f1 74 18 59 4c aa 	vmulps 0x4(%rdx,%rbp,4){1to4},%xmm1,%xmm1
    1291:	01 
    1292:	48 98                	cltq   
    1294:	4d 63 c9             	movslq %r9d,%r9
    1297:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
    129b:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
    12a0:	41 8d 44 2d 03       	lea    0x3(%r13,%rbp,1),%eax
    12a5:	c4 a3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%r9,4),%xmm1,%xmm1
    12ac:	45 8d 4c 2b 03       	lea    0x3(%r11,%rbp,1),%r9d
    12b1:	62 f1 74 18 59 4c aa 	vmulps 0x8(%rdx,%rbp,4){1to4},%xmm1,%xmm1
    12b8:	02 
    12b9:	48 98                	cltq   
    12bb:	4d 63 c9             	movslq %r9d,%r9
    12be:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
    12c2:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
    12c7:	41 8d 44 2d 04       	lea    0x4(%r13,%rbp,1),%eax
    12cc:	c4 a3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%r9,4),%xmm1,%xmm1
    12d3:	45 8d 4c 2b 04       	lea    0x4(%r11,%rbp,1),%r9d
    12d8:	62 f1 74 18 59 4c aa 	vmulps 0xc(%rdx,%rbp,4){1to4},%xmm1,%xmm1
    12df:	03 
    12e0:	48 98                	cltq   
    12e2:	4d 63 c9             	movslq %r9d,%r9
    12e5:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
    12e9:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
    12ee:	41 8d 44 2d 05       	lea    0x5(%r13,%rbp,1),%eax
    12f3:	c4 a3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%r9,4),%xmm1,%xmm1
    12fa:	45 8d 4c 2b 05       	lea    0x5(%r11,%rbp,1),%r9d
    12ff:	62 f1 74 18 59 4c aa 	vmulps 0x10(%rdx,%rbp,4){1to4},%xmm1,%xmm1
    1306:	04 
    1307:	48 98                	cltq   
    1309:	4d 63 c9             	movslq %r9d,%r9
    130c:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
    1310:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
    1315:	41 8d 44 2d 06       	lea    0x6(%r13,%rbp,1),%eax
    131a:	c4 a3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%r9,4),%xmm1,%xmm1
    1321:	45 8d 4c 2b 06       	lea    0x6(%r11,%rbp,1),%r9d
    1326:	62 f1 74 18 59 4c aa 	vmulps 0x14(%rdx,%rbp,4){1to4},%xmm1,%xmm1
    132d:	05 
    132e:	48 98                	cltq   
    1330:	4d 63 c9             	movslq %r9d,%r9
    1333:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
    1337:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
    133c:	41 8d 44 2d 07       	lea    0x7(%r13,%rbp,1),%eax
    1341:	c4 a3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%r9,4),%xmm1,%xmm1
    1348:	45 8d 4c 2b 07       	lea    0x7(%r11,%rbp,1),%r9d
    134d:	62 f1 74 18 59 4c aa 	vmulps 0x18(%rdx,%rbp,4){1to4},%xmm1,%xmm1
    1354:	06 
    1355:	48 98                	cltq   
    1357:	4d 63 c9             	movslq %r9d,%r9
    135a:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
    135e:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
    1363:	c4 a3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%r9,4),%xmm1,%xmm1
    136a:	62 f1 74 18 59 4c aa 	vmulps 0x1c(%rdx,%rbp,4){1to4},%xmm1,%xmm1
    1371:	07 
    1372:	48 83 c5 08          	add    $0x8,%rbp
    1376:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
    137a:	49 39 ee             	cmp    %rbp,%r14
    137d:	0f 85 bd fe ff ff    	jne    1240 <tile2+0xe0>
    1383:	48 85 db             	test   %rbx,%rbx
    1386:	0f 84 64 fe ff ff    	je     11f0 <tile2+0x90>
    138c:	41 8d 04 2b          	lea    (%r11,%rbp,1),%eax
    1390:	45 8d 4c 2d 00       	lea    0x0(%r13,%rbp,1),%r9d
    1395:	48 8d 2c aa          	lea    (%rdx,%rbp,4),%rbp
    1399:	45 31 d2             	xor    %r10d,%r10d
    139c:	0f 1f 40 00          	nopl   0x0(%rax)
    13a0:	43 8d 0c 11          	lea    (%r9,%r10,1),%ecx
    13a4:	46 8d 3c 10          	lea    (%rax,%r10,1),%r15d
    13a8:	48 63 c9             	movslq %ecx,%rcx
    13ab:	4d 63 ff             	movslq %r15d,%r15
    13ae:	c5 fa 10 0c 8e       	vmovss (%rsi,%rcx,4),%xmm1
    13b3:	c4 a3 71 21 0c be 10 	vinsertps $0x10,(%rsi,%r15,4),%xmm1,%xmm1
    13ba:	62 b1 74 18 59 4c 95 	vmulps 0x0(%rbp,%r10,4){1to4},%xmm1,%xmm1
    13c1:	00 
    13c2:	49 ff c2             	inc    %r10
    13c5:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
    13c9:	4c 39 d3             	cmp    %r10,%rbx
    13cc:	75 d2                	jne    13a0 <tile2+0x240>
    13ce:	e9 1d fe ff ff       	jmp    11f0 <tile2+0x90>
    13d3:	4c 8b 74 24 10       	mov    0x10(%rsp),%r14
    13d8:	44 89 f0             	mov    %r14d,%eax
    13db:	25 fe ff ff 7f       	and    $0x7ffffffe,%eax
    13e0:	44 39 f0             	cmp    %r14d,%eax
    13e3:	0f 8d 88 02 00 00    	jge    1671 <tile2+0x511>
    13e9:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
    13ee:	85 ed                	test   %ebp,%ebp
    13f0:	0f 8e 4f 02 00 00    	jle    1645 <tile2+0x4e5>
    13f6:	b9 01 1e 00 00       	mov    $0x1e01,%ecx
    13fb:	44 89 f0             	mov    %r14d,%eax
    13fe:	41 89 ea             	mov    %ebp,%r10d
    1401:	45 89 d3             	mov    %r10d,%r11d
    1404:	44 89 d3             	mov    %r10d,%ebx
    1407:	25 fe ff ff 7f       	and    $0x7ffffffe,%eax
    140c:	41 83 e3 07          	and    $0x7,%r11d
    1410:	81 e3 f8 ff ff 7f    	and    $0x7ffffff8,%ebx
    1416:	45 89 f1             	mov    %r14d,%r9d
    1419:	c4 c2 70 f7 ce       	bextr  %ecx,%r14d,%ecx
    141e:	0f af cd             	imul   %ebp,%ecx
    1421:	01 c9                	add    %ecx,%ecx
    1423:	eb 1f                	jmp    1444 <tile2+0x2e4>
    1425:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    142c:	00 00 00 00 
    1430:	c5 fa 11 04 87       	vmovss %xmm0,(%rdi,%rax,4)
    1435:	48 ff c0             	inc    %rax
    1438:	4c 01 d1             	add    %r10,%rcx
    143b:	4c 39 c8             	cmp    %r9,%rax
    143e:	0f 84 2d 02 00 00    	je     1671 <tile2+0x511>
    1444:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
    1448:	45 31 f6             	xor    %r14d,%r14d
    144b:	83 fd 08             	cmp    $0x8,%ebp
    144e:	0f 82 e8 00 00 00    	jb     153c <tile2+0x3dc>
    1454:	66 66 66 2e 0f 1f 84 	data16 data16 cs nopw 0x0(%rax,%rax,1)
    145b:	00 00 00 00 00 
    1460:	46 8d 04 31          	lea    (%rcx,%r14,1),%r8d
    1464:	4d 63 c0             	movslq %r8d,%r8
    1467:	c4 a1 7a 10 0c 86    	vmovss (%rsi,%r8,4),%xmm1
    146d:	46 8d 44 31 01       	lea    0x1(%rcx,%r14,1),%r8d
    1472:	c4 a1 72 59 0c b2    	vmulss (%rdx,%r14,4),%xmm1,%xmm1
    1478:	4d 63 c0             	movslq %r8d,%r8
    147b:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    147f:	c4 a1 7a 10 0c 86    	vmovss (%rsi,%r8,4),%xmm1
    1485:	46 8d 44 31 02       	lea    0x2(%rcx,%r14,1),%r8d
    148a:	c4 a1 72 59 4c b2 04 	vmulss 0x4(%rdx,%r14,4),%xmm1,%xmm1
    1491:	4d 63 c0             	movslq %r8d,%r8
    1494:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    1498:	c4 a1 7a 10 0c 86    	vmovss (%rsi,%r8,4),%xmm1
    149e:	46 8d 44 31 03       	lea    0x3(%rcx,%r14,1),%r8d
    14a3:	c4 a1 72 59 4c b2 08 	vmulss 0x8(%rdx,%r14,4),%xmm1,%xmm1
    14aa:	4d 63 c0             	movslq %r8d,%r8
    14ad:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    14b1:	c4 a1 7a 10 0c 86    	vmovss (%rsi,%r8,4),%xmm1
    14b7:	46 8d 44 31 04       	lea    0x4(%rcx,%r14,1),%r8d
    14bc:	c4 a1 72 59 4c b2 0c 	vmulss 0xc(%rdx,%r14,4),%xmm1,%xmm1
    14c3:	4d 63 c0             	movslq %r8d,%r8
    14c6:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    14ca:	c4 a1 7a 10 0c 86    	vmovss (%rsi,%r8,4),%xmm1
    14d0:	46 8d 44 31 05       	lea    0x5(%rcx,%r14,1),%r8d
    14d5:	c4 a1 72 59 4c b2 10 	vmulss 0x10(%rdx,%r14,4),%xmm1,%xmm1
    14dc:	4d 63 c0             	movslq %r8d,%r8
    14df:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    14e3:	c4 a1 7a 10 0c 86    	vmovss (%rsi,%r8,4),%xmm1
    14e9:	46 8d 44 31 06       	lea    0x6(%rcx,%r14,1),%r8d
    14ee:	c4 a1 72 59 4c b2 14 	vmulss 0x14(%rdx,%r14,4),%xmm1,%xmm1
    14f5:	4d 63 c0             	movslq %r8d,%r8
    14f8:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    14fc:	c4 a1 7a 10 0c 86    	vmovss (%rsi,%r8,4),%xmm1
    1502:	46 8d 44 31 07       	lea    0x7(%rcx,%r14,1),%r8d
    1507:	c4 a1 72 59 4c b2 18 	vmulss 0x18(%rdx,%r14,4),%xmm1,%xmm1
    150e:	4d 63 c0             	movslq %r8d,%r8
    1511:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    1515:	c4 a1 7a 10 0c 86    	vmovss (%rsi,%r8,4),%xmm1
    151b:	c4 a1 72 59 4c b2 1c 	vmulss 0x1c(%rdx,%r14,4),%xmm1,%xmm1
    1522:	49 83 c6 08          	add    $0x8,%r14
    1526:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    152a:	4c 39 f3             	cmp    %r14,%rbx
    152d:	0f 85 2d ff ff ff    	jne    1460 <tile2+0x300>
    1533:	4d 85 db             	test   %r11,%r11
    1536:	0f 84 f4 fe ff ff    	je     1430 <tile2+0x2d0>
    153c:	4e 8d 3c b2          	lea    (%rdx,%r14,4),%r15
    1540:	41 01 ce             	add    %ecx,%r14d
    1543:	45 31 e4             	xor    %r12d,%r12d
    1546:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    154d:	00 00 00 
    1550:	47 8d 04 26          	lea    (%r14,%r12,1),%r8d
    1554:	4d 63 c0             	movslq %r8d,%r8
    1557:	c4 a1 7a 10 0c 86    	vmovss (%rsi,%r8,4),%xmm1
    155d:	c4 81 72 59 0c a7    	vmulss (%r15,%r12,4),%xmm1,%xmm1
    1563:	49 ff c4             	inc    %r12
    1566:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    156a:	4d 39 e3             	cmp    %r12,%r11
    156d:	75 e1                	jne    1550 <tile2+0x3f0>
    156f:	e9 bc fe ff ff       	jmp    1430 <tile2+0x2d0>
    1574:	4c 8b 74 24 10       	mov    0x10(%rsp),%r14
    1579:	41 83 fe 10          	cmp    $0x10,%r14d
    157d:	0f 83 02 01 00 00    	jae    1685 <tile2+0x525>
    1583:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
    1588:	31 d2                	xor    %edx,%edx
    158a:	48 89 ce             	mov    %rcx,%rsi
    158d:	48 83 e6 07          	and    $0x7,%rsi
    1591:	48 89 d0             	mov    %rdx,%rax
    1594:	74 1a                	je     15b0 <tile2+0x450>
    1596:	48 89 d0             	mov    %rdx,%rax
    1599:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    15a0:	48 c7 04 c7 00 00 00 	movq   $0x0,(%rdi,%rax,8)
    15a7:	00 
    15a8:	48 ff c0             	inc    %rax
    15ab:	48 ff ce             	dec    %rsi
    15ae:	75 f0                	jne    15a0 <tile2+0x440>
    15b0:	48 29 ca             	sub    %rcx,%rdx
    15b3:	48 83 fa f8          	cmp    $0xfffffffffffffff8,%rdx
    15b7:	77 7f                	ja     1638 <tile2+0x4d8>
    15b9:	8d 54 00 0e          	lea    0xe(%rax,%rax,1),%edx
    15bd:	48 29 c1             	sub    %rax,%rcx
    15c0:	8d 42 f2             	lea    -0xe(%rdx),%eax
    15c3:	44 8d 42 f4          	lea    -0xc(%rdx),%r8d
    15c7:	8d 72 f6             	lea    -0xa(%rdx),%esi
    15ca:	48 98                	cltq   
    15cc:	48 c7 04 87 00 00 00 	movq   $0x0,(%rdi,%rax,4)
    15d3:	00 
    15d4:	49 63 c0             	movslq %r8d,%rax
    15d7:	44 8d 42 f8          	lea    -0x8(%rdx),%r8d
    15db:	48 c7 04 87 00 00 00 	movq   $0x0,(%rdi,%rax,4)
    15e2:	00 
    15e3:	48 63 c6             	movslq %esi,%rax
    15e6:	8d 72 fa             	lea    -0x6(%rdx),%esi
    15e9:	48 c7 04 87 00 00 00 	movq   $0x0,(%rdi,%rax,4)
    15f0:	00 
    15f1:	49 63 c0             	movslq %r8d,%rax
    15f4:	44 8d 42 fc          	lea    -0x4(%rdx),%r8d
    15f8:	48 c7 04 87 00 00 00 	movq   $0x0,(%rdi,%rax,4)
    15ff:	00 
    1600:	48 63 c6             	movslq %esi,%rax
    1603:	8d 72 fe             	lea    -0x2(%rdx),%esi
    1606:	48 63 d2             	movslq %edx,%rdx
    1609:	48 c7 04 87 00 00 00 	movq   $0x0,(%rdi,%rax,4)
    1610:	00 
    1611:	49 63 c0             	movslq %r8d,%rax
    1614:	48 c7 04 87 00 00 00 	movq   $0x0,(%rdi,%rax,4)
    161b:	00 
    161c:	48 63 c6             	movslq %esi,%rax
    161f:	48 c7 04 87 00 00 00 	movq   $0x0,(%rdi,%rax,4)
    1626:	00 
    1627:	48 c7 04 97 00 00 00 	movq   $0x0,(%rdi,%rdx,4)
    162e:	00 
    162f:	83 c2 10             	add    $0x10,%edx
    1632:	48 83 c1 f8          	add    $0xfffffffffffffff8,%rcx
    1636:	75 88                	jne    15c0 <tile2+0x460>
    1638:	44 89 f0             	mov    %r14d,%eax
    163b:	25 fe ff ff 7f       	and    $0x7ffffffe,%eax
    1640:	44 39 f0             	cmp    %r14d,%eax
    1643:	74 2c                	je     1671 <tile2+0x511>
    1645:	8b 4c 24 04          	mov    0x4(%rsp),%ecx
    1649:	48 8b 5c 24 18       	mov    0x18(%rsp),%rbx
    164e:	f7 d0                	not    %eax
    1650:	31 f6                	xor    %esi,%esi
    1652:	41 01 c6             	add    %eax,%r14d
    1655:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    165c:	00 00 00 
    165f:	4a 8d 14 b5 04 00 00 	lea    0x4(,%r14,4),%rdx
    1666:	00 
    1667:	48 8d 3c cf          	lea    (%rdi,%rcx,8),%rdi
    166b:	c5 f8 77             	vzeroupper 
    166e:	ff 14 03             	call   *(%rbx,%rax,1)
    1671:	31 c0                	xor    %eax,%eax
    1673:	48 83 c4 28          	add    $0x28,%rsp
    1677:	5b                   	pop    %rbx
    1678:	41 5c                	pop    %r12
    167a:	41 5d                	pop    %r13
    167c:	41 5e                	pop    %r14
    167e:	41 5f                	pop    %r15
    1680:	5d                   	pop    %rbp
    1681:	c5 f8 77             	vzeroupper 
    1684:	c3                   	ret    
    1685:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
    168a:	31 d2                	xor    %edx,%edx
    168c:	48 8d 44 cf f8       	lea    -0x8(%rdi,%rcx,8),%rax
    1691:	48 39 f8             	cmp    %rdi,%rax
    1694:	0f 82 f0 fe ff ff    	jb     158a <tile2+0x42a>
    169a:	48 8d 41 ff          	lea    -0x1(%rcx),%rax
    169e:	48 be 00 00 00 40 ff 	movabs $0xffffffff40000000,%rsi
    16a5:	ff ff ff 
    16a8:	48 21 c6             	and    %rax,%rsi
    16ab:	0f 85 d9 fe ff ff    	jne    158a <tile2+0x42a>
    16b1:	4c 8d 44 cf fc       	lea    -0x4(%rdi,%rcx,8),%r8
    16b6:	48 8d 77 04          	lea    0x4(%rdi),%rsi
    16ba:	49 39 f0             	cmp    %rsi,%r8
    16bd:	0f 82 c7 fe ff ff    	jb     158a <tile2+0x42a>
    16c3:	48 c1 e8 3d          	shr    $0x3d,%rax
    16c7:	0f 85 bd fe ff ff    	jne    158a <tile2+0x42a>
    16cd:	41 81 fe 80 00 00 00 	cmp    $0x80,%r14d
    16d4:	73 07                	jae    16dd <tile2+0x57d>
    16d6:	31 d2                	xor    %edx,%edx
    16d8:	e9 7f 00 00 00       	jmp    175c <tile2+0x5fc>
    16dd:	b8 07 18 00 00       	mov    $0x1807,%eax
    16e2:	89 ca                	mov    %ecx,%edx
    16e4:	81 e2 c0 ff ff 3f    	and    $0x3fffffc0,%edx
    16ea:	31 f6                	xor    %esi,%esi
    16ec:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
    16f0:	c4 c2 78 f7 c6       	bextr  %eax,%r14d,%eax
    16f5:	48 c1 e0 09          	shl    $0x9,%rax
    16f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    1700:	62 f1 7c 48 11 44 37 	vmovups %zmm0,0x40(%rdi,%rsi,1)
    1707:	01 
    1708:	62 f1 7c 48 11 04 37 	vmovups %zmm0,(%rdi,%rsi,1)
    170f:	62 f1 7c 48 11 44 37 	vmovups %zmm0,0xc0(%rdi,%rsi,1)
    1716:	03 
    1717:	62 f1 7c 48 11 44 37 	vmovups %zmm0,0x80(%rdi,%rsi,1)
    171e:	02 
    171f:	62 f1 7c 48 11 44 37 	vmovups %zmm0,0x140(%rdi,%rsi,1)
    1726:	05 
    1727:	62 f1 7c 48 11 44 37 	vmovups %zmm0,0x100(%rdi,%rsi,1)
    172e:	04 
    172f:	62 f1 7c 48 11 44 37 	vmovups %zmm0,0x1c0(%rdi,%rsi,1)
    1736:	07 
    1737:	62 f1 7c 48 11 44 37 	vmovups %zmm0,0x180(%rdi,%rsi,1)
    173e:	06 
    173f:	48 81 c6 00 02 00 00 	add    $0x200,%rsi
    1746:	48 39 f0             	cmp    %rsi,%rax
    1749:	75 b5                	jne    1700 <tile2+0x5a0>
    174b:	39 ca                	cmp    %ecx,%edx
    174d:	0f 84 e5 fe ff ff    	je     1638 <tile2+0x4d8>
    1753:	f6 c1 38             	test   $0x38,%cl
    1756:	0f 84 2e fe ff ff    	je     158a <tile2+0x42a>
    175c:	48 89 d0             	mov    %rdx,%rax
    175f:	89 ca                	mov    %ecx,%edx
    1761:	81 e2 f8 ff ff 3f    	and    $0x3ffffff8,%edx
    1767:	48 89 c6             	mov    %rax,%rsi
    176a:	01 c0                	add    %eax,%eax
    176c:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
    1770:	48 29 d6             	sub    %rdx,%rsi
    1773:	66 66 66 66 2e 0f 1f 	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
    177a:	84 00 00 00 00 00 
    1780:	48 98                	cltq   
    1782:	62 f1 7c 48 11 04 87 	vmovups %zmm0,(%rdi,%rax,4)
    1789:	83 c0 10             	add    $0x10,%eax
    178c:	48 83 c6 08          	add    $0x8,%rsi
    1790:	75 ee                	jne    1780 <tile2+0x620>
    1792:	39 ca                	cmp    %ecx,%edx
    1794:	0f 85 f0 fd ff ff    	jne    158a <tile2+0x42a>
    179a:	e9 99 fe ff ff       	jmp    1638 <tile2+0x4d8>
    179f:	90                   	nop

00000000000017a0 <baseline>:
    17a0:	48 8d 05 f9 ff ff ff 	lea    -0x7(%rip),%rax        # 17a0 <baseline>
    17a7:	49 b9 00 00 00 00 00 	movabs $0x0,%r9
    17ae:	00 00 00 
    17b1:	41 89 ca             	mov    %ecx,%r10d
    17b4:	49 01 c1             	add    %rax,%r9
    17b7:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    17bc:	45 09 c2             	or     %r8d,%r10d
    17bf:	0f 88 81 01 00 00    	js     1946 <baseline+0x1a6>
    17c5:	85 c9                	test   %ecx,%ecx
    17c7:	0f 84 77 01 00 00    	je     1944 <baseline+0x1a4>
    17cd:	55                   	push   %rbp
    17ce:	41 57                	push   %r15
    17d0:	41 56                	push   %r14
    17d2:	41 55                	push   %r13
    17d4:	41 54                	push   %r12
    17d6:	53                   	push   %rbx
    17d7:	50                   	push   %rax
    17d8:	89 c8                	mov    %ecx,%eax
    17da:	45 85 c0             	test   %r8d,%r8d
    17dd:	0f 84 64 01 00 00    	je     1947 <baseline+0x1a7>
    17e3:	44 89 c1             	mov    %r8d,%ecx
    17e6:	41 89 c9             	mov    %ecx,%r9d
    17e9:	41 89 ca             	mov    %ecx,%r10d
    17ec:	41 83 e1 07          	and    $0x7,%r9d
    17f0:	41 81 e2 f8 ff ff 7f 	and    $0x7ffffff8,%r10d
    17f7:	45 31 db             	xor    %r11d,%r11d
    17fa:	31 db                	xor    %ebx,%ebx
    17fc:	eb 16                	jmp    1814 <baseline+0x74>
    17fe:	66 90                	xchg   %ax,%ax
    1800:	c5 fa 11 04 9f       	vmovss %xmm0,(%rdi,%rbx,4)
    1805:	48 ff c3             	inc    %rbx
    1808:	49 01 cb             	add    %rcx,%r11
    180b:	48 39 c3             	cmp    %rax,%rbx
    180e:	0f 84 4a 01 00 00    	je     195e <baseline+0x1be>
    1814:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
    1818:	45 31 f6             	xor    %r14d,%r14d
    181b:	41 83 f8 08          	cmp    $0x8,%r8d
    181f:	0f 82 e7 00 00 00    	jb     190c <baseline+0x16c>
    1825:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    182c:	00 00 00 00 
    1830:	43 8d 2c 33          	lea    (%r11,%r14,1),%ebp
    1834:	4c 63 fd             	movslq %ebp,%r15
    1837:	43 8d 6c 33 01       	lea    0x1(%r11,%r14,1),%ebp
    183c:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
    1842:	c4 a1 72 59 0c b2    	vmulss (%rdx,%r14,4),%xmm1,%xmm1
    1848:	4c 63 fd             	movslq %ebp,%r15
    184b:	43 8d 6c 33 02       	lea    0x2(%r11,%r14,1),%ebp
    1850:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    1854:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
    185a:	4c 63 fd             	movslq %ebp,%r15
    185d:	43 8d 6c 33 03       	lea    0x3(%r11,%r14,1),%ebp
    1862:	c4 a1 72 59 4c b2 04 	vmulss 0x4(%rdx,%r14,4),%xmm1,%xmm1
    1869:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    186d:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
    1873:	4c 63 fd             	movslq %ebp,%r15
    1876:	43 8d 6c 33 04       	lea    0x4(%r11,%r14,1),%ebp
    187b:	c4 a1 72 59 4c b2 08 	vmulss 0x8(%rdx,%r14,4),%xmm1,%xmm1
    1882:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    1886:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
    188c:	4c 63 fd             	movslq %ebp,%r15
    188f:	43 8d 6c 33 05       	lea    0x5(%r11,%r14,1),%ebp
    1894:	c4 a1 72 59 4c b2 0c 	vmulss 0xc(%rdx,%r14,4),%xmm1,%xmm1
    189b:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    189f:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
    18a5:	4c 63 fd             	movslq %ebp,%r15
    18a8:	43 8d 6c 33 06       	lea    0x6(%r11,%r14,1),%ebp
    18ad:	c4 a1 72 59 4c b2 10 	vmulss 0x10(%rdx,%r14,4),%xmm1,%xmm1
    18b4:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    18b8:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
    18be:	4c 63 fd             	movslq %ebp,%r15
    18c1:	43 8d 6c 33 07       	lea    0x7(%r11,%r14,1),%ebp
    18c6:	c4 a1 72 59 4c b2 14 	vmulss 0x14(%rdx,%r14,4),%xmm1,%xmm1
    18cd:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    18d1:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
    18d7:	4c 63 fd             	movslq %ebp,%r15
    18da:	c4 a1 72 59 4c b2 18 	vmulss 0x18(%rdx,%r14,4),%xmm1,%xmm1
    18e1:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    18e5:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
    18eb:	c4 a1 72 59 4c b2 1c 	vmulss 0x1c(%rdx,%r14,4),%xmm1,%xmm1
    18f2:	49 83 c6 08          	add    $0x8,%r14
    18f6:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    18fa:	4d 39 f2             	cmp    %r14,%r10
    18fd:	0f 85 2d ff ff ff    	jne    1830 <baseline+0x90>
    1903:	4d 85 c9             	test   %r9,%r9
    1906:	0f 84 f4 fe ff ff    	je     1800 <baseline+0x60>
    190c:	4e 8d 3c b2          	lea    (%rdx,%r14,4),%r15
    1910:	45 01 de             	add    %r11d,%r14d
    1913:	45 31 e4             	xor    %r12d,%r12d
    1916:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    191d:	00 00 00 
    1920:	43 8d 2c 26          	lea    (%r14,%r12,1),%ebp
    1924:	4c 63 ed             	movslq %ebp,%r13
    1927:	c4 a1 7a 10 0c ae    	vmovss (%rsi,%r13,4),%xmm1
    192d:	c4 81 72 59 0c a7    	vmulss (%r15,%r12,4),%xmm1,%xmm1
    1933:	49 ff c4             	inc    %r12
    1936:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    193a:	4d 39 e1             	cmp    %r12,%r9
    193d:	75 e1                	jne    1920 <baseline+0x180>
    193f:	e9 bc fe ff ff       	jmp    1800 <baseline+0x60>
    1944:	31 c0                	xor    %eax,%eax
    1946:	c3                   	ret    
    1947:	48 c1 e0 02          	shl    $0x2,%rax
    194b:	48 b9 00 00 00 00 00 	movabs $0x0,%rcx
    1952:	00 00 00 
    1955:	31 f6                	xor    %esi,%esi
    1957:	48 89 c2             	mov    %rax,%rdx
    195a:	41 ff 14 09          	call   *(%r9,%rcx,1)
    195e:	31 c0                	xor    %eax,%eax
    1960:	48 83 c4 08          	add    $0x8,%rsp
    1964:	5b                   	pop    %rbx
    1965:	41 5c                	pop    %r12
    1967:	41 5d                	pop    %r13
    1969:	41 5e                	pop    %r14
    196b:	41 5f                	pop    %r15
    196d:	5d                   	pop    %rbp
    196e:	c3                   	ret    
    196f:	90                   	nop

0000000000001970 <_mlir_integrated>:
    1970:	53                   	push   %rbx
    1971:	48 89 fb             	mov    %rdi,%rbx
    1974:	48 8b 53 10          	mov    0x10(%rbx),%rdx
    1978:	4c 8b 53 20          	mov    0x20(%rbx),%r10
    197c:	48 8b 37             	mov    (%rdi),%rsi
    197f:	48 8b 4f 08          	mov    0x8(%rdi),%rcx
    1983:	4c 8b 5b 18          	mov    0x18(%rbx),%r11
    1987:	48 8d 05 f9 ff ff ff 	lea    -0x7(%rip),%rax        # 1987 <_mlir_integrated+0x17>
    198e:	49 b9 00 00 00 00 00 	movabs $0x0,%r9
    1995:	00 00 00 
    1998:	49 01 c1             	add    %rax,%r9
    199b:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    19a2:	00 00 00 
    19a5:	48 8b 3e             	mov    (%rsi),%rdi
    19a8:	48 8b 31             	mov    (%rcx),%rsi
    19ab:	48 8b 12             	mov    (%rdx),%rdx
    19ae:	41 8b 0b             	mov    (%r11),%ecx
    19b1:	45 8b 02             	mov    (%r10),%r8d
    19b4:	41 ff 14 01          	call   *(%r9,%rax,1)
    19b8:	48 8b 4b 28          	mov    0x28(%rbx),%rcx
    19bc:	89 01                	mov    %eax,(%rcx)
    19be:	5b                   	pop    %rbx
    19bf:	c3                   	ret    

00000000000019c0 <_mlir_tile8>:
    19c0:	53                   	push   %rbx
    19c1:	48 89 fb             	mov    %rdi,%rbx
    19c4:	48 8b 53 10          	mov    0x10(%rbx),%rdx
    19c8:	4c 8b 53 20          	mov    0x20(%rbx),%r10
    19cc:	48 8b 37             	mov    (%rdi),%rsi
    19cf:	48 8b 4f 08          	mov    0x8(%rdi),%rcx
    19d3:	4c 8b 5b 18          	mov    0x18(%rbx),%r11
    19d7:	48 8d 05 f9 ff ff ff 	lea    -0x7(%rip),%rax        # 19d7 <_mlir_tile8+0x17>
    19de:	49 b9 00 00 00 00 00 	movabs $0x0,%r9
    19e5:	00 00 00 
    19e8:	49 01 c1             	add    %rax,%r9
    19eb:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    19f2:	00 00 00 
    19f5:	48 8b 3e             	mov    (%rsi),%rdi
    19f8:	48 8b 31             	mov    (%rcx),%rsi
    19fb:	48 8b 12             	mov    (%rdx),%rdx
    19fe:	41 8b 0b             	mov    (%r11),%ecx
    1a01:	45 8b 02             	mov    (%r10),%r8d
    1a04:	41 ff 14 01          	call   *(%r9,%rax,1)
    1a08:	48 8b 4b 28          	mov    0x28(%rbx),%rcx
    1a0c:	89 01                	mov    %eax,(%rcx)
    1a0e:	5b                   	pop    %rbx
    1a0f:	c3                   	ret    

0000000000001a10 <_mlir_tile4>:
    1a10:	53                   	push   %rbx
    1a11:	48 89 fb             	mov    %rdi,%rbx
    1a14:	48 8b 53 10          	mov    0x10(%rbx),%rdx
    1a18:	4c 8b 53 20          	mov    0x20(%rbx),%r10
    1a1c:	48 8b 37             	mov    (%rdi),%rsi
    1a1f:	48 8b 4f 08          	mov    0x8(%rdi),%rcx
    1a23:	4c 8b 5b 18          	mov    0x18(%rbx),%r11
    1a27:	48 8d 05 f9 ff ff ff 	lea    -0x7(%rip),%rax        # 1a27 <_mlir_tile4+0x17>
    1a2e:	49 b9 00 00 00 00 00 	movabs $0x0,%r9
    1a35:	00 00 00 
    1a38:	49 01 c1             	add    %rax,%r9
    1a3b:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    1a42:	00 00 00 
    1a45:	48 8b 3e             	mov    (%rsi),%rdi
    1a48:	48 8b 31             	mov    (%rcx),%rsi
    1a4b:	48 8b 12             	mov    (%rdx),%rdx
    1a4e:	41 8b 0b             	mov    (%r11),%ecx
    1a51:	45 8b 02             	mov    (%r10),%r8d
    1a54:	41 ff 14 01          	call   *(%r9,%rax,1)
    1a58:	48 8b 4b 28          	mov    0x28(%rbx),%rcx
    1a5c:	89 01                	mov    %eax,(%rcx)
    1a5e:	5b                   	pop    %rbx
    1a5f:	c3                   	ret    

0000000000001a60 <_mlir_tile2>:
    1a60:	53                   	push   %rbx
    1a61:	48 89 fb             	mov    %rdi,%rbx
    1a64:	48 8b 53 10          	mov    0x10(%rbx),%rdx
    1a68:	4c 8b 53 20          	mov    0x20(%rbx),%r10
    1a6c:	48 8b 37             	mov    (%rdi),%rsi
    1a6f:	48 8b 4f 08          	mov    0x8(%rdi),%rcx
    1a73:	4c 8b 5b 18          	mov    0x18(%rbx),%r11
    1a77:	48 8d 05 f9 ff ff ff 	lea    -0x7(%rip),%rax        # 1a77 <_mlir_tile2+0x17>
    1a7e:	49 b9 00 00 00 00 00 	movabs $0x0,%r9
    1a85:	00 00 00 
    1a88:	49 01 c1             	add    %rax,%r9
    1a8b:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    1a92:	00 00 00 
    1a95:	48 8b 3e             	mov    (%rsi),%rdi
    1a98:	48 8b 31             	mov    (%rcx),%rsi
    1a9b:	48 8b 12             	mov    (%rdx),%rdx
    1a9e:	41 8b 0b             	mov    (%r11),%ecx
    1aa1:	45 8b 02             	mov    (%r10),%r8d
    1aa4:	41 ff 14 01          	call   *(%r9,%rax,1)
    1aa8:	48 8b 4b 28          	mov    0x28(%rbx),%rcx
    1aac:	89 01                	mov    %eax,(%rcx)
    1aae:	5b                   	pop    %rbx
    1aaf:	c3                   	ret    

0000000000001ab0 <_mlir_baseline>:
    1ab0:	55                   	push   %rbp
    1ab1:	41 57                	push   %r15
    1ab3:	41 56                	push   %r14
    1ab5:	41 55                	push   %r13
    1ab7:	41 54                	push   %r12
    1ab9:	53                   	push   %rbx
    1aba:	50                   	push   %rax
    1abb:	48 8d 05 f9 ff ff ff 	lea    -0x7(%rip),%rax        # 1abb <_mlir_baseline+0xb>
    1ac2:	49 b8 00 00 00 00 00 	movabs $0x0,%r8
    1ac9:	00 00 00 
    1acc:	48 8b 57 18          	mov    0x18(%rdi),%rdx
    1ad0:	bb ff ff ff ff       	mov    $0xffffffff,%ebx
    1ad5:	49 01 c0             	add    %rax,%r8
    1ad8:	48 8b 47 20          	mov    0x20(%rdi),%rax
    1adc:	8b 12                	mov    (%rdx),%edx
    1ade:	8b 08                	mov    (%rax),%ecx
    1ae0:	89 d0                	mov    %edx,%eax
    1ae2:	09 c8                	or     %ecx,%eax
    1ae4:	0f 88 ad 01 00 00    	js     1c97 <_mlir_baseline+0x1e7>
    1aea:	48 85 d2             	test   %rdx,%rdx
    1aed:	0f 84 81 01 00 00    	je     1c74 <_mlir_baseline+0x1c4>
    1af3:	48 8b 07             	mov    (%rdi),%rax
    1af6:	89 c9                	mov    %ecx,%ecx
    1af8:	48 8b 00             	mov    (%rax),%rax
    1afb:	48 85 c9             	test   %rcx,%rcx
    1afe:	0f 84 74 01 00 00    	je     1c78 <_mlir_baseline+0x1c8>
    1b04:	48 8b 77 08          	mov    0x8(%rdi),%rsi
    1b08:	4c 8b 47 10          	mov    0x10(%rdi),%r8
    1b0c:	41 89 c9             	mov    %ecx,%r9d
    1b0f:	41 89 ca             	mov    %ecx,%r10d
    1b12:	41 83 e1 07          	and    $0x7,%r9d
    1b16:	41 81 e2 f8 ff ff 7f 	and    $0x7ffffff8,%r10d
    1b1d:	45 31 db             	xor    %r11d,%r11d
    1b20:	31 db                	xor    %ebx,%ebx
    1b22:	48 8b 36             	mov    (%rsi),%rsi
    1b25:	4d 8b 00             	mov    (%r8),%r8
    1b28:	eb 1a                	jmp    1b44 <_mlir_baseline+0x94>
    1b2a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1b30:	c5 fa 11 04 98       	vmovss %xmm0,(%rax,%rbx,4)
    1b35:	48 ff c3             	inc    %rbx
    1b38:	49 01 cb             	add    %rcx,%r11
    1b3b:	48 39 d3             	cmp    %rdx,%rbx
    1b3e:	0f 84 30 01 00 00    	je     1c74 <_mlir_baseline+0x1c4>
    1b44:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
    1b48:	45 31 f6             	xor    %r14d,%r14d
    1b4b:	83 f9 08             	cmp    $0x8,%ecx
    1b4e:	0f 82 e8 00 00 00    	jb     1c3c <_mlir_baseline+0x18c>
    1b54:	66 66 66 2e 0f 1f 84 	data16 data16 cs nopw 0x0(%rax,%rax,1)
    1b5b:	00 00 00 00 00 
    1b60:	43 8d 2c 33          	lea    (%r11,%r14,1),%ebp
    1b64:	4c 63 fd             	movslq %ebp,%r15
    1b67:	43 8d 6c 33 01       	lea    0x1(%r11,%r14,1),%ebp
    1b6c:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
    1b72:	c4 81 72 59 0c b0    	vmulss (%r8,%r14,4),%xmm1,%xmm1
    1b78:	4c 63 fd             	movslq %ebp,%r15
    1b7b:	43 8d 6c 33 02       	lea    0x2(%r11,%r14,1),%ebp
    1b80:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    1b84:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
    1b8a:	4c 63 fd             	movslq %ebp,%r15
    1b8d:	43 8d 6c 33 03       	lea    0x3(%r11,%r14,1),%ebp
    1b92:	c4 81 72 59 4c b0 04 	vmulss 0x4(%r8,%r14,4),%xmm1,%xmm1
    1b99:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    1b9d:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
    1ba3:	4c 63 fd             	movslq %ebp,%r15
    1ba6:	43 8d 6c 33 04       	lea    0x4(%r11,%r14,1),%ebp
    1bab:	c4 81 72 59 4c b0 08 	vmulss 0x8(%r8,%r14,4),%xmm1,%xmm1
    1bb2:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    1bb6:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
    1bbc:	4c 63 fd             	movslq %ebp,%r15
    1bbf:	43 8d 6c 33 05       	lea    0x5(%r11,%r14,1),%ebp
    1bc4:	c4 81 72 59 4c b0 0c 	vmulss 0xc(%r8,%r14,4),%xmm1,%xmm1
    1bcb:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    1bcf:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
    1bd5:	4c 63 fd             	movslq %ebp,%r15
    1bd8:	43 8d 6c 33 06       	lea    0x6(%r11,%r14,1),%ebp
    1bdd:	c4 81 72 59 4c b0 10 	vmulss 0x10(%r8,%r14,4),%xmm1,%xmm1
    1be4:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    1be8:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
    1bee:	4c 63 fd             	movslq %ebp,%r15
    1bf1:	43 8d 6c 33 07       	lea    0x7(%r11,%r14,1),%ebp
    1bf6:	c4 81 72 59 4c b0 14 	vmulss 0x14(%r8,%r14,4),%xmm1,%xmm1
    1bfd:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    1c01:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
    1c07:	4c 63 fd             	movslq %ebp,%r15
    1c0a:	c4 81 72 59 4c b0 18 	vmulss 0x18(%r8,%r14,4),%xmm1,%xmm1
    1c11:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    1c15:	c4 a1 7a 10 0c be    	vmovss (%rsi,%r15,4),%xmm1
    1c1b:	c4 81 72 59 4c b0 1c 	vmulss 0x1c(%r8,%r14,4),%xmm1,%xmm1
    1c22:	49 83 c6 08          	add    $0x8,%r14
    1c26:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    1c2a:	4d 39 f2             	cmp    %r14,%r10
    1c2d:	0f 85 2d ff ff ff    	jne    1b60 <_mlir_baseline+0xb0>
    1c33:	4d 85 c9             	test   %r9,%r9
    1c36:	0f 84 f4 fe ff ff    	je     1b30 <_mlir_baseline+0x80>
    1c3c:	4f 8d 3c b0          	lea    (%r8,%r14,4),%r15
    1c40:	45 01 de             	add    %r11d,%r14d
    1c43:	45 31 e4             	xor    %r12d,%r12d
    1c46:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    1c4d:	00 00 00 
    1c50:	43 8d 2c 26          	lea    (%r14,%r12,1),%ebp
    1c54:	4c 63 ed             	movslq %ebp,%r13
    1c57:	c4 a1 7a 10 0c ae    	vmovss (%rsi,%r13,4),%xmm1
    1c5d:	c4 81 72 59 0c a7    	vmulss (%r15,%r12,4),%xmm1,%xmm1
    1c63:	49 ff c4             	inc    %r12
    1c66:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    1c6a:	4d 39 e1             	cmp    %r12,%r9
    1c6d:	75 e1                	jne    1c50 <_mlir_baseline+0x1a0>
    1c6f:	e9 bc fe ff ff       	jmp    1b30 <_mlir_baseline+0x80>
    1c74:	31 db                	xor    %ebx,%ebx
    1c76:	eb 1f                	jmp    1c97 <_mlir_baseline+0x1e7>
    1c78:	48 c1 e2 02          	shl    $0x2,%rdx
    1c7c:	48 b9 00 00 00 00 00 	movabs $0x0,%rcx
    1c83:	00 00 00 
    1c86:	49 89 fe             	mov    %rdi,%r14
    1c89:	31 db                	xor    %ebx,%ebx
    1c8b:	48 89 c7             	mov    %rax,%rdi
    1c8e:	31 f6                	xor    %esi,%esi
    1c90:	41 ff 14 08          	call   *(%r8,%rcx,1)
    1c94:	4c 89 f7             	mov    %r14,%rdi
    1c97:	48 8b 47 28          	mov    0x28(%rdi),%rax
    1c9b:	89 18                	mov    %ebx,(%rax)
    1c9d:	48 83 c4 08          	add    $0x8,%rsp
    1ca1:	5b                   	pop    %rbx
    1ca2:	41 5c                	pop    %r12
    1ca4:	41 5d                	pop    %r13
    1ca6:	41 5e                	pop    %r14
    1ca8:	41 5f                	pop    %r15
    1caa:	5d                   	pop    %rbp
    1cab:	c3                   	ret    
