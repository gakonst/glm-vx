0000000000005390 <glm_vx_matvec>:
    5390:	55                   	push   %rbp
    5391:	41 57                	push   %r15
    5393:	41 56                	push   %r14
    5395:	41 55                	push   %r13
    5397:	41 54                	push   %r12
    5399:	53                   	push   %rbx
    539a:	48 83 ec 38          	sub    $0x38,%rsp
    539e:	48 8d 05 f9 ff ff ff 	lea    -0x7(%rip),%rax        # 539e <glm_vx_matvec+0xe>
    53a5:	49 bd 00 00 00 00 00 	movabs $0x0,%r13
    53ac:	00 00 00 
    53af:	bb ff ff ff ff       	mov    $0xffffffff,%ebx
    53b4:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
    53b9:	48 89 0c 24          	mov    %rcx,(%rsp)
    53bd:	49 01 c5             	add    %rax,%r13
    53c0:	89 c8                	mov    %ecx,%eax
    53c2:	44 09 c0             	or     %r8d,%eax
    53c5:	0f 88 5a 07 00 00    	js     5b25 <glm_vx_matvec+0x795>
    53cb:	48 8b 04 24          	mov    (%rsp),%rax
    53cf:	41 89 c2             	mov    %eax,%r10d
    53d2:	41 c1 ea 03          	shr    $0x3,%r10d
    53d6:	0f 84 35 02 00 00    	je     5611 <glm_vx_matvec+0x281>
    53dc:	44 89 d0             	mov    %r10d,%eax
    53df:	45 85 c0             	test   %r8d,%r8d
    53e2:	0f 84 96 04 00 00    	je     587e <glm_vx_matvec+0x4ee>
    53e8:	48 b9 00 00 00 00 00 	movabs $0x0,%rcx
    53ef:	00 00 00 
    53f2:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
    53f7:	45 89 c1             	mov    %r8d,%r9d
    53fa:	45 89 cb             	mov    %r9d,%r11d
    53fd:	62 d2 7d 28 7c c0    	vpbroadcastd %r8d,%ymm0
    5403:	41 83 e3 07          	and    $0x7,%r11d
    5407:	41 81 e1 f8 ff ff 7f 	and    $0x7ffffff8,%r9d
    540e:	31 db                	xor    %ebx,%ebx
    5410:	c4 c1 7d 6f 4c 0d 00 	vmovdqa 0x0(%r13,%rcx,1),%ymm1
    5417:	eb 23                	jmp    543c <glm_vx_matvec+0xac>
    5419:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    5420:	48 89 d9             	mov    %rbx,%rcx
    5423:	48 c1 e1 23          	shl    $0x23,%rcx
    5427:	48 ff c3             	inc    %rbx
    542a:	48 c1 f9 1e          	sar    $0x1e,%rcx
    542e:	c5 fc 11 1c 0f       	vmovups %ymm3,(%rdi,%rcx,1)
    5433:	48 39 c3             	cmp    %rax,%rbx
    5436:	0f 84 d5 01 00 00    	je     5611 <glm_vx_matvec+0x281>
    543c:	8d 0c dd 00 00 00 00 	lea    0x0(,%rbx,8),%ecx
    5443:	62 f2 7d 28 7c d1    	vpbroadcastd %ecx,%ymm2
    5449:	c5 ed eb d1          	vpor   %ymm1,%ymm2,%ymm2
    544d:	c4 e2 7d 40 d2       	vpmulld %ymm2,%ymm0,%ymm2
    5452:	41 83 f8 08          	cmp    $0x8,%r8d
    5456:	73 18                	jae    5470 <glm_vx_matvec+0xe0>
    5458:	45 31 f6             	xor    %r14d,%r14d
    545b:	c5 e0 57 db          	vxorps %xmm3,%xmm3,%xmm3
    545f:	e9 75 01 00 00       	jmp    55d9 <glm_vx_matvec+0x249>
    5464:	66 66 66 2e 0f 1f 84 	data16 data16 cs nopw 0x0(%rax,%rax,1)
    546b:	00 00 00 00 00 
    5470:	c5 e0 57 db          	vxorps %xmm3,%xmm3,%xmm3
    5474:	45 31 f6             	xor    %r14d,%r14d
    5477:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    547e:	00 00 
    5480:	62 d2 7d 28 7c e6    	vpbroadcastd %r14d,%ymm4
    5486:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
    548a:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
    548e:	41 8d 4e 01          	lea    0x1(%r14),%ecx
    5492:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
    5496:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
    549d:	62 b1 54 38 59 24 b2 	vmulps (%rdx,%r14,4){1to8},%ymm5,%ymm4
    54a4:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
    54a8:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
    54ac:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
    54b0:	62 f2 7d 28 7c e1    	vpbroadcastd %ecx,%ymm4
    54b6:	41 8d 4e 02          	lea    0x2(%r14),%ecx
    54ba:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
    54be:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
    54c5:	62 b1 54 38 59 64 b2 	vmulps 0x4(%rdx,%r14,4){1to8},%ymm5,%ymm4
    54cc:	01 
    54cd:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
    54d1:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
    54d5:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
    54d9:	62 f2 7d 28 7c e1    	vpbroadcastd %ecx,%ymm4
    54df:	41 8d 4e 03          	lea    0x3(%r14),%ecx
    54e3:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
    54e7:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
    54ee:	62 b1 54 38 59 64 b2 	vmulps 0x8(%rdx,%r14,4){1to8},%ymm5,%ymm4
    54f5:	02 
    54f6:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
    54fa:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
    54fe:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
    5502:	62 f2 7d 28 7c e1    	vpbroadcastd %ecx,%ymm4
    5508:	41 8d 4e 04          	lea    0x4(%r14),%ecx
    550c:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
    5510:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
    5517:	62 b1 54 38 59 64 b2 	vmulps 0xc(%rdx,%r14,4){1to8},%ymm5,%ymm4
    551e:	03 
    551f:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
    5523:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
    5527:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
    552b:	62 f2 7d 28 7c e1    	vpbroadcastd %ecx,%ymm4
    5531:	41 8d 4e 05          	lea    0x5(%r14),%ecx
    5535:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
    5539:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
    5540:	62 b1 54 38 59 64 b2 	vmulps 0x10(%rdx,%r14,4){1to8},%ymm5,%ymm4
    5547:	04 
    5548:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
    554c:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
    5550:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
    5554:	62 f2 7d 28 7c e1    	vpbroadcastd %ecx,%ymm4
    555a:	41 8d 4e 06          	lea    0x6(%r14),%ecx
    555e:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
    5562:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
    5569:	62 b1 54 38 59 64 b2 	vmulps 0x14(%rdx,%r14,4){1to8},%ymm5,%ymm4
    5570:	05 
    5571:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
    5575:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
    5579:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
    557d:	62 f2 7d 28 7c e1    	vpbroadcastd %ecx,%ymm4
    5583:	41 8d 4e 07          	lea    0x7(%r14),%ecx
    5587:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
    558b:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
    5592:	62 b1 54 38 59 64 b2 	vmulps 0x18(%rdx,%r14,4){1to8},%ymm5,%ymm4
    5599:	06 
    559a:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
    559e:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
    55a2:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
    55a6:	62 f2 7d 28 7c e1    	vpbroadcastd %ecx,%ymm4
    55ac:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
    55b0:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
    55b7:	62 b1 54 38 59 64 b2 	vmulps 0x1c(%rdx,%r14,4){1to8},%ymm5,%ymm4
    55be:	07 
    55bf:	49 83 c6 08          	add    $0x8,%r14
    55c3:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
    55c7:	4d 39 ce             	cmp    %r9,%r14
    55ca:	0f 85 b0 fe ff ff    	jne    5480 <glm_vx_matvec+0xf0>
    55d0:	4d 85 db             	test   %r11,%r11
    55d3:	0f 84 47 fe ff ff    	je     5420 <glm_vx_matvec+0x90>
    55d9:	4c 89 d9             	mov    %r11,%rcx
    55dc:	0f 1f 40 00          	nopl   0x0(%rax)
    55e0:	62 d2 7d 28 7c e6    	vpbroadcastd %r14d,%ymm4
    55e6:	c5 fd 46 c8          	kxnorb %k0,%k0,%k1
    55ea:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
    55ee:	c5 ed fe e4          	vpaddd %ymm4,%ymm2,%ymm4
    55f2:	62 f2 7d 29 92 2c a6 	vgatherdps (%rsi,%ymm4,4),%ymm5{%k1}
    55f9:	62 b1 54 38 59 24 b2 	vmulps (%rdx,%r14,4){1to8},%ymm5,%ymm4
    5600:	49 ff c6             	inc    %r14
    5603:	48 ff c9             	dec    %rcx
    5606:	c5 e4 58 dc          	vaddps %ymm4,%ymm3,%ymm3
    560a:	75 d4                	jne    55e0 <glm_vx_matvec+0x250>
    560c:	e9 0f fe ff ff       	jmp    5420 <glm_vx_matvec+0x90>
    5611:	4c 8b 0c 24          	mov    (%rsp),%r9
    5615:	44 89 cd             	mov    %r9d,%ebp
    5618:	41 d1 e9             	shr    %r9d
    561b:	81 e5 f8 ff ff 7f    	and    $0x7ffffff8,%ebp
    5621:	44 89 cb             	mov    %r9d,%ebx
    5624:	83 e3 03             	and    $0x3,%ebx
    5627:	0f 84 4a 02 00 00    	je     5877 <glm_vx_matvec+0x4e7>
    562d:	45 85 c0             	test   %r8d,%r8d
    5630:	0f 8e f5 02 00 00    	jle    592b <glm_vx_matvec+0x59b>
    5636:	b9 03 1c 00 00       	mov    $0x1c03,%ecx
    563b:	44 8d 65 01          	lea    0x1(%rbp),%r12d
    563f:	89 e8                	mov    %ebp,%eax
    5641:	48 89 6c 24 18       	mov    %rbp,0x18(%rsp)
    5646:	45 89 c3             	mov    %r8d,%r11d
    5649:	45 89 df             	mov    %r11d,%r15d
    564c:	41 83 e7 07          	and    $0x7,%r15d
    5650:	41 81 e3 f8 ff ff 7f 	and    $0x7ffffff8,%r11d
    5657:	43 8d 3c 00          	lea    (%r8,%r8,1),%edi
    565b:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    5660:	89 d8                	mov    %ebx,%eax
    5662:	48 89 5c 24 10       	mov    %rbx,0x10(%rsp)
    5667:	4c 89 6c 24 20       	mov    %r13,0x20(%rsp)
    566c:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    5671:	31 db                	xor    %ebx,%ebx
    5673:	c4 e2 70 f7 2c 24    	bextr  %ecx,(%rsp),%ebp
    5679:	45 0f af e0          	imul   %r8d,%r12d
    567d:	41 0f af e8          	imul   %r8d,%ebp
    5681:	c1 e5 03             	shl    $0x3,%ebp
    5684:	eb 32                	jmp    56b8 <glm_vx_matvec+0x328>
    5686:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    568d:	00 00 00 
    5690:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
    5695:	4c 8b 4c 24 08       	mov    0x8(%rsp),%r9
    569a:	49 01 fc             	add    %rdi,%r12
    569d:	48 01 fd             	add    %rdi,%rbp
    56a0:	48 8d 0c 58          	lea    (%rax,%rbx,2),%rcx
    56a4:	48 ff c3             	inc    %rbx
    56a7:	c4 c1 78 13 04 89    	vmovlps %xmm0,(%r9,%rcx,4)
    56ad:	48 3b 5c 24 28       	cmp    0x28(%rsp),%rbx
    56b2:	0f 84 ab 01 00 00    	je     5863 <glm_vx_matvec+0x4d3>
    56b8:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    56bc:	45 31 f6             	xor    %r14d,%r14d
    56bf:	41 83 f8 08          	cmp    $0x8,%r8d
    56c3:	0f 82 53 01 00 00    	jb     581c <glm_vx_matvec+0x48c>
    56c9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    56d0:	42 8d 44 35 00       	lea    0x0(%rbp,%r14,1),%eax
    56d5:	43 8d 0c 34          	lea    (%r12,%r14,1),%ecx
    56d9:	48 98                	cltq   
    56db:	48 63 c9             	movslq %ecx,%rcx
    56de:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
    56e3:	42 8d 44 35 01       	lea    0x1(%rbp,%r14,1),%eax
    56e8:	c4 e3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%rcx,4),%xmm1,%xmm1
    56ef:	43 8d 4c 34 01       	lea    0x1(%r12,%r14,1),%ecx
    56f4:	62 b1 74 18 59 0c b2 	vmulps (%rdx,%r14,4){1to4},%xmm1,%xmm1
    56fb:	48 98                	cltq   
    56fd:	48 63 c9             	movslq %ecx,%rcx
    5700:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
    5704:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
    5709:	42 8d 44 35 02       	lea    0x2(%rbp,%r14,1),%eax
    570e:	c4 e3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%rcx,4),%xmm1,%xmm1
    5715:	43 8d 4c 34 02       	lea    0x2(%r12,%r14,1),%ecx
    571a:	62 b1 74 18 59 4c b2 	vmulps 0x4(%rdx,%r14,4){1to4},%xmm1,%xmm1
    5721:	01 
    5722:	48 98                	cltq   
    5724:	48 63 c9             	movslq %ecx,%rcx
    5727:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
    572b:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
    5730:	42 8d 44 35 03       	lea    0x3(%rbp,%r14,1),%eax
    5735:	c4 e3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%rcx,4),%xmm1,%xmm1
    573c:	43 8d 4c 34 03       	lea    0x3(%r12,%r14,1),%ecx
    5741:	62 b1 74 18 59 4c b2 	vmulps 0x8(%rdx,%r14,4){1to4},%xmm1,%xmm1
    5748:	02 
    5749:	48 98                	cltq   
    574b:	48 63 c9             	movslq %ecx,%rcx
    574e:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
    5752:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
    5757:	42 8d 44 35 04       	lea    0x4(%rbp,%r14,1),%eax
    575c:	c4 e3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%rcx,4),%xmm1,%xmm1
    5763:	43 8d 4c 34 04       	lea    0x4(%r12,%r14,1),%ecx
    5768:	62 b1 74 18 59 4c b2 	vmulps 0xc(%rdx,%r14,4){1to4},%xmm1,%xmm1
    576f:	03 
    5770:	48 98                	cltq   
    5772:	48 63 c9             	movslq %ecx,%rcx
    5775:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
    5779:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
    577e:	42 8d 44 35 05       	lea    0x5(%rbp,%r14,1),%eax
    5783:	c4 e3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%rcx,4),%xmm1,%xmm1
    578a:	43 8d 4c 34 05       	lea    0x5(%r12,%r14,1),%ecx
    578f:	62 b1 74 18 59 4c b2 	vmulps 0x10(%rdx,%r14,4){1to4},%xmm1,%xmm1
    5796:	04 
    5797:	48 98                	cltq   
    5799:	48 63 c9             	movslq %ecx,%rcx
    579c:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
    57a0:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
    57a5:	42 8d 44 35 06       	lea    0x6(%rbp,%r14,1),%eax
    57aa:	c4 e3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%rcx,4),%xmm1,%xmm1
    57b1:	43 8d 4c 34 06       	lea    0x6(%r12,%r14,1),%ecx
    57b6:	62 b1 74 18 59 4c b2 	vmulps 0x14(%rdx,%r14,4){1to4},%xmm1,%xmm1
    57bd:	05 
    57be:	48 98                	cltq   
    57c0:	48 63 c9             	movslq %ecx,%rcx
    57c3:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
    57c7:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
    57cc:	42 8d 44 35 07       	lea    0x7(%rbp,%r14,1),%eax
    57d1:	c4 e3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%rcx,4),%xmm1,%xmm1
    57d8:	43 8d 4c 34 07       	lea    0x7(%r12,%r14,1),%ecx
    57dd:	62 b1 74 18 59 4c b2 	vmulps 0x18(%rdx,%r14,4){1to4},%xmm1,%xmm1
    57e4:	06 
    57e5:	48 98                	cltq   
    57e7:	48 63 c9             	movslq %ecx,%rcx
    57ea:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
    57ee:	c5 fa 10 0c 86       	vmovss (%rsi,%rax,4),%xmm1
    57f3:	c4 e3 71 21 0c 8e 10 	vinsertps $0x10,(%rsi,%rcx,4),%xmm1,%xmm1
    57fa:	62 b1 74 18 59 4c b2 	vmulps 0x1c(%rdx,%r14,4){1to4},%xmm1,%xmm1
    5801:	07 
    5802:	49 83 c6 08          	add    $0x8,%r14
    5806:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
    580a:	4d 39 f3             	cmp    %r14,%r11
    580d:	0f 85 bd fe ff ff    	jne    56d0 <glm_vx_matvec+0x340>
    5813:	4d 85 ff             	test   %r15,%r15
    5816:	0f 84 74 fe ff ff    	je     5690 <glm_vx_matvec+0x300>
    581c:	47 8d 0c 34          	lea    (%r12,%r14,1),%r9d
    5820:	42 8d 4c 35 00       	lea    0x0(%rbp,%r14,1),%ecx
    5825:	4e 8d 34 b2          	lea    (%rdx,%r14,4),%r14
    5829:	31 c0                	xor    %eax,%eax
    582b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    5830:	44 8d 14 01          	lea    (%rcx,%rax,1),%r10d
    5834:	45 8d 2c 01          	lea    (%r9,%rax,1),%r13d
    5838:	4d 63 d2             	movslq %r10d,%r10
    583b:	4d 63 ed             	movslq %r13d,%r13
    583e:	c4 a1 7a 10 0c 96    	vmovss (%rsi,%r10,4),%xmm1
    5844:	c4 a3 71 21 0c ae 10 	vinsertps $0x10,(%rsi,%r13,4),%xmm1,%xmm1
    584b:	62 d1 74 18 59 0c 86 	vmulps (%r14,%rax,4){1to4},%xmm1,%xmm1
    5852:	48 ff c0             	inc    %rax
    5855:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
    5859:	49 39 c7             	cmp    %rax,%r15
    585c:	75 d2                	jne    5830 <glm_vx_matvec+0x4a0>
    585e:	e9 2d fe ff ff       	jmp    5690 <glm_vx_matvec+0x300>
    5863:	4c 8b 6c 24 20       	mov    0x20(%rsp),%r13
    5868:	48 8b 6c 24 18       	mov    0x18(%rsp),%rbp
    586d:	48 8b 5c 24 10       	mov    0x10(%rsp),%rbx
    5872:	e9 f7 00 00 00       	jmp    596e <glm_vx_matvec+0x5de>
    5877:	31 db                	xor    %ebx,%ebx
    5879:	e9 f0 00 00 00       	jmp    596e <glm_vx_matvec+0x5de>
    587e:	41 89 c1             	mov    %eax,%r9d
    5881:	41 83 e1 07          	and    $0x7,%r9d
    5885:	83 3c 24 40          	cmpl   $0x40,(%rsp)
    5889:	73 05                	jae    5890 <glm_vx_matvec+0x500>
    588b:	45 31 db             	xor    %r11d,%r11d
    588e:	eb 5d                	jmp    58ed <glm_vx_matvec+0x55d>
    5890:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    5895:	25 f8 ff ff 0f       	and    $0xffffff8,%eax
    589a:	45 31 db             	xor    %r11d,%r11d
    589d:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    58a1:	48 81 c1 e0 00 00 00 	add    $0xe0,%rcx
    58a8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    58af:	00 
    58b0:	62 f1 fe 48 7f 81 20 	vmovdqu64 %zmm0,-0xe0(%rcx)
    58b7:	ff ff ff 
    58ba:	62 f1 fe 48 7f 81 60 	vmovdqu64 %zmm0,-0xa0(%rcx)
    58c1:	ff ff ff 
    58c4:	62 f1 fe 48 7f 81 a0 	vmovdqu64 %zmm0,-0x60(%rcx)
    58cb:	ff ff ff 
    58ce:	62 f1 fe 48 7f 81 e0 	vmovdqu64 %zmm0,-0x20(%rcx)
    58d5:	ff ff ff 
    58d8:	49 83 c3 08          	add    $0x8,%r11
    58dc:	48 81 c1 00 01 00 00 	add    $0x100,%rcx
    58e3:	4c 39 d8             	cmp    %r11,%rax
    58e6:	75 c8                	jne    58b0 <glm_vx_matvec+0x520>
    58e8:	4d 85 c9             	test   %r9,%r9
    58eb:	74 22                	je     590f <glm_vx_matvec+0x57f>
    58ed:	49 c1 e3 05          	shl    $0x5,%r11
    58f1:	4c 03 5c 24 08       	add    0x8(%rsp),%r11
    58f6:	41 c1 e1 05          	shl    $0x5,%r9d
    58fa:	31 c0                	xor    %eax,%eax
    58fc:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    5900:	c4 c1 7e 7f 04 03    	vmovdqu %ymm0,(%r11,%rax,1)
    5906:	48 83 c0 20          	add    $0x20,%rax
    590a:	49 39 c1             	cmp    %rax,%r9
    590d:	75 f1                	jne    5900 <glm_vx_matvec+0x570>
    590f:	4c 8b 0c 24          	mov    (%rsp),%r9
    5913:	44 89 cd             	mov    %r9d,%ebp
    5916:	41 d1 e9             	shr    %r9d
    5919:	81 e5 f8 ff ff 7f    	and    $0x7ffffff8,%ebp
    591f:	44 89 cb             	mov    %r9d,%ebx
    5922:	83 e3 03             	and    $0x3,%ebx
    5925:	0f 84 bc 01 00 00    	je     5ae7 <glm_vx_matvec+0x757>
    592b:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    5930:	41 c1 e1 03          	shl    $0x3,%r9d
    5934:	44 89 d0             	mov    %r10d,%eax
    5937:	48 c1 e0 05          	shl    $0x5,%rax
    593b:	49 89 f7             	mov    %rsi,%r15
    593e:	49 89 d6             	mov    %rdx,%r14
    5941:	4d 89 c4             	mov    %r8,%r12
    5944:	31 f6                	xor    %esi,%esi
    5946:	41 83 e1 18          	and    $0x18,%r9d
    594a:	4c 89 ca             	mov    %r9,%rdx
    594d:	48 01 c8             	add    %rcx,%rax
    5950:	48 b9 00 00 00 00 00 	movabs $0x0,%rcx
    5957:	00 00 00 
    595a:	48 89 c7             	mov    %rax,%rdi
    595d:	c5 f8 77             	vzeroupper 
    5960:	41 ff 54 0d 00       	call   *0x0(%r13,%rcx,1)
    5965:	4d 89 e0             	mov    %r12,%r8
    5968:	4c 89 f2             	mov    %r14,%rdx
    596b:	4c 89 fe             	mov    %r15,%rsi
    596e:	44 8d 4c 5d 00       	lea    0x0(%rbp,%rbx,2),%r9d
    5973:	44 3b 0c 24          	cmp    (%rsp),%r9d
    5977:	0f 8d a6 01 00 00    	jge    5b23 <glm_vx_matvec+0x793>
    597d:	45 85 c0             	test   %r8d,%r8d
    5980:	0f 8e 5d 01 00 00    	jle    5ae3 <glm_vx_matvec+0x753>
    5986:	8b 0c 24             	mov    (%rsp),%ecx
    5989:	44 89 c8             	mov    %r9d,%eax
    598c:	45 0f af c8          	imul   %r8d,%r9d
    5990:	45 89 c2             	mov    %r8d,%r10d
    5993:	45 89 d3             	mov    %r10d,%r11d
    5996:	44 89 d3             	mov    %r10d,%ebx
    5999:	41 83 e3 07          	and    $0x7,%r11d
    599d:	81 e3 f8 ff ff 7f    	and    $0x7ffffff8,%ebx
    59a3:	eb 24                	jmp    59c9 <glm_vx_matvec+0x639>
    59a5:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    59ac:	00 00 00 00 
    59b0:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
    59b5:	4d 01 d1             	add    %r10,%r9
    59b8:	c5 fa 11 04 87       	vmovss %xmm0,(%rdi,%rax,4)
    59bd:	48 ff c0             	inc    %rax
    59c0:	48 39 c8             	cmp    %rcx,%rax
    59c3:	0f 83 5a 01 00 00    	jae    5b23 <glm_vx_matvec+0x793>
    59c9:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    59cd:	45 31 f6             	xor    %r14d,%r14d
    59d0:	41 83 f8 08          	cmp    $0x8,%r8d
    59d4:	0f 82 da 00 00 00    	jb     5ab4 <glm_vx_matvec+0x724>
    59da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    59e0:	43 8d 3c 31          	lea    (%r9,%r14,1),%edi
    59e4:	48 63 ff             	movslq %edi,%rdi
    59e7:	c5 fa 10 0c be       	vmovss (%rsi,%rdi,4),%xmm1
    59ec:	43 8d 7c 31 01       	lea    0x1(%r9,%r14,1),%edi
    59f1:	c4 a1 72 59 0c b2    	vmulss (%rdx,%r14,4),%xmm1,%xmm1
    59f7:	48 63 ff             	movslq %edi,%rdi
    59fa:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    59fe:	c5 fa 10 0c be       	vmovss (%rsi,%rdi,4),%xmm1
    5a03:	43 8d 7c 31 02       	lea    0x2(%r9,%r14,1),%edi
    5a08:	c4 a1 72 59 4c b2 04 	vmulss 0x4(%rdx,%r14,4),%xmm1,%xmm1
    5a0f:	48 63 ff             	movslq %edi,%rdi
    5a12:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    5a16:	c5 fa 10 0c be       	vmovss (%rsi,%rdi,4),%xmm1
    5a1b:	43 8d 7c 31 03       	lea    0x3(%r9,%r14,1),%edi
    5a20:	c4 a1 72 59 4c b2 08 	vmulss 0x8(%rdx,%r14,4),%xmm1,%xmm1
    5a27:	48 63 ff             	movslq %edi,%rdi
    5a2a:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    5a2e:	c5 fa 10 0c be       	vmovss (%rsi,%rdi,4),%xmm1
    5a33:	43 8d 7c 31 04       	lea    0x4(%r9,%r14,1),%edi
    5a38:	c4 a1 72 59 4c b2 0c 	vmulss 0xc(%rdx,%r14,4),%xmm1,%xmm1
    5a3f:	48 63 ff             	movslq %edi,%rdi
    5a42:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    5a46:	c5 fa 10 0c be       	vmovss (%rsi,%rdi,4),%xmm1
    5a4b:	43 8d 7c 31 05       	lea    0x5(%r9,%r14,1),%edi
    5a50:	c4 a1 72 59 4c b2 10 	vmulss 0x10(%rdx,%r14,4),%xmm1,%xmm1
    5a57:	48 63 ff             	movslq %edi,%rdi
    5a5a:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    5a5e:	c5 fa 10 0c be       	vmovss (%rsi,%rdi,4),%xmm1
    5a63:	43 8d 7c 31 06       	lea    0x6(%r9,%r14,1),%edi
    5a68:	c4 a1 72 59 4c b2 14 	vmulss 0x14(%rdx,%r14,4),%xmm1,%xmm1
    5a6f:	48 63 ff             	movslq %edi,%rdi
    5a72:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    5a76:	c5 fa 10 0c be       	vmovss (%rsi,%rdi,4),%xmm1
    5a7b:	43 8d 7c 31 07       	lea    0x7(%r9,%r14,1),%edi
    5a80:	c4 a1 72 59 4c b2 18 	vmulss 0x18(%rdx,%r14,4),%xmm1,%xmm1
    5a87:	48 63 ff             	movslq %edi,%rdi
    5a8a:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    5a8e:	c5 fa 10 0c be       	vmovss (%rsi,%rdi,4),%xmm1
    5a93:	c4 a1 72 59 4c b2 1c 	vmulss 0x1c(%rdx,%r14,4),%xmm1,%xmm1
    5a9a:	49 83 c6 08          	add    $0x8,%r14
    5a9e:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    5aa2:	4c 39 f3             	cmp    %r14,%rbx
    5aa5:	0f 85 35 ff ff ff    	jne    59e0 <glm_vx_matvec+0x650>
    5aab:	4d 85 db             	test   %r11,%r11
    5aae:	0f 84 fc fe ff ff    	je     59b0 <glm_vx_matvec+0x620>
    5ab4:	4e 8d 3c b2          	lea    (%rdx,%r14,4),%r15
    5ab8:	45 01 ce             	add    %r9d,%r14d
    5abb:	45 31 e4             	xor    %r12d,%r12d
    5abe:	66 90                	xchg   %ax,%ax
    5ac0:	43 8d 3c 26          	lea    (%r14,%r12,1),%edi
    5ac4:	48 63 ff             	movslq %edi,%rdi
    5ac7:	c5 fa 10 0c be       	vmovss (%rsi,%rdi,4),%xmm1
    5acc:	c4 81 72 59 0c a7    	vmulss (%r15,%r12,4),%xmm1,%xmm1
    5ad2:	49 ff c4             	inc    %r12
    5ad5:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
    5ad9:	4d 39 e3             	cmp    %r12,%r11
    5adc:	75 e2                	jne    5ac0 <glm_vx_matvec+0x730>
    5ade:	e9 cd fe ff ff       	jmp    59b0 <glm_vx_matvec+0x620>
    5ae3:	01 db                	add    %ebx,%ebx
    5ae5:	eb 0a                	jmp    5af1 <glm_vx_matvec+0x761>
    5ae7:	31 db                	xor    %ebx,%ebx
    5ae9:	41 89 e9             	mov    %ebp,%r9d
    5aec:	3b 2c 24             	cmp    (%rsp),%ebp
    5aef:	74 34                	je     5b25 <glm_vx_matvec+0x795>
    5af1:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    5af6:	44 89 c9             	mov    %r9d,%ecx
    5af9:	f7 d3                	not    %ebx
    5afb:	31 f6                	xor    %esi,%esi
    5afd:	48 8d 3c 88          	lea    (%rax,%rcx,4),%rdi
    5b01:	48 8b 04 24          	mov    (%rsp),%rax
    5b05:	01 d8                	add    %ebx,%eax
    5b07:	29 e8                	sub    %ebp,%eax
    5b09:	48 8d 14 85 04 00 00 	lea    0x4(,%rax,4),%rdx
    5b10:	00 
    5b11:	48 b8 00 00 00 00 00 	movabs $0x0,%rax
    5b18:	00 00 00 
    5b1b:	c5 f8 77             	vzeroupper 
    5b1e:	41 ff 54 05 00       	call   *0x0(%r13,%rax,1)
    5b23:	31 db                	xor    %ebx,%ebx
    5b25:	89 d8                	mov    %ebx,%eax
    5b27:	48 83 c4 38          	add    $0x38,%rsp
    5b2b:	5b                   	pop    %rbx
    5b2c:	41 5c                	pop    %r12
    5b2e:	41 5d                	pop    %r13
    5b30:	41 5e                	pop    %r14
    5b32:	41 5f                	pop    %r15
    5b34:	5d                   	pop    %rbp
    5b35:	c5 f8 77             	vzeroupper 
    5b38:	c3                   	ret    
    5b39:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
