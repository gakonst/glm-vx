
evidence/vx-row-tile/types.o:     file format elf64-x86-64


Disassembly of section .ltext:

0000000000000000 <vector_add>:
   0:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
   4:	c3                   	ret    
   5:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
   c:	00 00 00 00 

0000000000000010 <half_to_float>:
  10:	85 d2                	test   %edx,%edx
  12:	0f 8e 13 01 00 00    	jle    12b <half_to_float+0x11b>
  18:	48 63 c2             	movslq %edx,%rax
  1b:	83 fa 08             	cmp    $0x8,%edx
  1e:	72 1e                	jb     3e <half_to_float+0x2e>
  20:	4c 8d 04 46          	lea    (%rsi,%rax,2),%r8
  24:	48 8d 0c 87          	lea    (%rdi,%rax,4),%rcx
  28:	4c 39 c7             	cmp    %r8,%rdi
  2b:	41 0f 92 c0          	setb   %r8b
  2f:	48 39 ce             	cmp    %rcx,%rsi
  32:	0f 92 c1             	setb   %cl
  35:	41 84 c8             	test   %cl,%r8b
  38:	0f 84 f3 00 00 00    	je     131 <half_to_float+0x121>
  3e:	31 c9                	xor    %ecx,%ecx
  40:	49 89 c8             	mov    %rcx,%r8
  43:	f6 c2 07             	test   $0x7,%dl
  46:	74 22                	je     6a <half_to_float+0x5a>
  48:	83 e2 07             	and    $0x7,%edx
  4b:	49 89 c8             	mov    %rcx,%r8
  4e:	66 90                	xchg   %ax,%ax
  50:	c4 a1 79 c4 04 46 00 	vpinsrw $0x0,(%rsi,%r8,2),%xmm0,%xmm0
  57:	c4 e2 79 13 c0       	vcvtph2ps %xmm0,%xmm0
  5c:	c4 a1 79 7e 04 87    	vmovd  %xmm0,(%rdi,%r8,4)
  62:	49 ff c0             	inc    %r8
  65:	48 ff ca             	dec    %rdx
  68:	75 e6                	jne    50 <half_to_float+0x40>
  6a:	48 29 c1             	sub    %rax,%rcx
  6d:	48 83 f9 f8          	cmp    $0xfffffffffffffff8,%rcx
  71:	0f 87 b4 00 00 00    	ja     12b <half_to_float+0x11b>
  77:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  7e:	00 00 
  80:	c4 a1 79 c4 04 46 00 	vpinsrw $0x0,(%rsi,%r8,2),%xmm0,%xmm0
  87:	c4 e2 79 13 c0       	vcvtph2ps %xmm0,%xmm0
  8c:	c4 a1 79 7e 04 87    	vmovd  %xmm0,(%rdi,%r8,4)
  92:	c4 a1 79 c4 44 46 02 	vpinsrw $0x0,0x2(%rsi,%r8,2),%xmm0,%xmm0
  99:	00 
  9a:	c4 e2 79 13 c0       	vcvtph2ps %xmm0,%xmm0
  9f:	c4 a1 79 7e 44 87 04 	vmovd  %xmm0,0x4(%rdi,%r8,4)
  a6:	c4 a1 79 c4 44 46 04 	vpinsrw $0x0,0x4(%rsi,%r8,2),%xmm0,%xmm0
  ad:	00 
  ae:	c4 e2 79 13 c0       	vcvtph2ps %xmm0,%xmm0
  b3:	c4 a1 79 7e 44 87 08 	vmovd  %xmm0,0x8(%rdi,%r8,4)
  ba:	c4 a1 79 c4 44 46 06 	vpinsrw $0x0,0x6(%rsi,%r8,2),%xmm0,%xmm0
  c1:	00 
  c2:	c4 e2 79 13 c0       	vcvtph2ps %xmm0,%xmm0
  c7:	c4 a1 79 7e 44 87 0c 	vmovd  %xmm0,0xc(%rdi,%r8,4)
  ce:	c4 a1 79 c4 44 46 08 	vpinsrw $0x0,0x8(%rsi,%r8,2),%xmm0,%xmm0
  d5:	00 
  d6:	c4 e2 79 13 c0       	vcvtph2ps %xmm0,%xmm0
  db:	c4 a1 79 7e 44 87 10 	vmovd  %xmm0,0x10(%rdi,%r8,4)
  e2:	c4 a1 79 c4 44 46 0a 	vpinsrw $0x0,0xa(%rsi,%r8,2),%xmm0,%xmm0
  e9:	00 
  ea:	c4 e2 79 13 c0       	vcvtph2ps %xmm0,%xmm0
  ef:	c4 a1 79 7e 44 87 14 	vmovd  %xmm0,0x14(%rdi,%r8,4)
  f6:	c4 a1 79 c4 44 46 0c 	vpinsrw $0x0,0xc(%rsi,%r8,2),%xmm0,%xmm0
  fd:	00 
  fe:	c4 e2 79 13 c0       	vcvtph2ps %xmm0,%xmm0
 103:	c4 a1 79 7e 44 87 18 	vmovd  %xmm0,0x18(%rdi,%r8,4)
 10a:	c4 a1 79 c4 44 46 0e 	vpinsrw $0x0,0xe(%rsi,%r8,2),%xmm0,%xmm0
 111:	00 
 112:	c4 e2 79 13 c0       	vcvtph2ps %xmm0,%xmm0
 117:	c4 a1 79 7e 44 87 1c 	vmovd  %xmm0,0x1c(%rdi,%r8,4)
 11e:	49 83 c0 08          	add    $0x8,%r8
 122:	4c 39 c0             	cmp    %r8,%rax
 125:	0f 85 55 ff ff ff    	jne    80 <half_to_float+0x70>
 12b:	31 c0                	xor    %eax,%eax
 12d:	c5 f8 77             	vzeroupper 
 130:	c3                   	ret    
 131:	83 fa 40             	cmp    $0x40,%edx
 134:	73 04                	jae    13a <half_to_float+0x12a>
 136:	31 c9                	xor    %ecx,%ecx
 138:	eb 6a                	jmp    1a4 <half_to_float+0x194>
 13a:	41 89 d0             	mov    %edx,%r8d
 13d:	41 c1 e8 06          	shr    $0x6,%r8d
 141:	89 d1                	mov    %edx,%ecx
 143:	81 e1 c0 ff ff 7f    	and    $0x7fffffc0,%ecx
 149:	45 31 c9             	xor    %r9d,%r9d
 14c:	41 c1 e0 07          	shl    $0x7,%r8d
 150:	62 b2 7d 48 13 04 0e 	vcvtph2ps (%rsi,%r9,1),%zmm0
 157:	62 b2 7d 48 13 4c 0e 	vcvtph2ps 0x20(%rsi,%r9,1),%zmm1
 15e:	01 
 15f:	62 b2 7d 48 13 54 0e 	vcvtph2ps 0x40(%rsi,%r9,1),%zmm2
 166:	02 
 167:	62 b2 7d 48 13 5c 0e 	vcvtph2ps 0x60(%rsi,%r9,1),%zmm3
 16e:	03 
 16f:	62 b1 fe 48 7f 04 4f 	vmovdqu64 %zmm0,(%rdi,%r9,2)
 176:	62 b1 7c 48 11 4c 4f 	vmovups %zmm1,0x40(%rdi,%r9,2)
 17d:	01 
 17e:	62 b1 7c 48 11 54 4f 	vmovups %zmm2,0x80(%rdi,%r9,2)
 185:	02 
 186:	62 b1 7c 48 11 5c 4f 	vmovups %zmm3,0xc0(%rdi,%r9,2)
 18d:	03 
 18e:	49 83 e9 80          	sub    $0xffffffffffffff80,%r9
 192:	4d 39 c8             	cmp    %r9,%r8
 195:	75 b9                	jne    150 <half_to_float+0x140>
 197:	39 d1                	cmp    %edx,%ecx
 199:	74 90                	je     12b <half_to_float+0x11b>
 19b:	f6 c2 38             	test   $0x38,%dl
 19e:	0f 84 9c fe ff ff    	je     40 <half_to_float+0x30>
 1a4:	49 89 c8             	mov    %rcx,%r8
 1a7:	89 d1                	mov    %edx,%ecx
 1a9:	81 e1 f8 ff ff 7f    	and    $0x7ffffff8,%ecx
 1af:	90                   	nop
 1b0:	c4 a2 7d 13 04 46    	vcvtph2ps (%rsi,%r8,2),%ymm0
 1b6:	c4 a1 7e 7f 04 87    	vmovdqu %ymm0,(%rdi,%r8,4)
 1bc:	49 83 c0 08          	add    $0x8,%r8
 1c0:	4c 39 c1             	cmp    %r8,%rcx
 1c3:	75 eb                	jne    1b0 <half_to_float+0x1a0>
 1c5:	39 d1                	cmp    %edx,%ecx
 1c7:	0f 84 5e ff ff ff    	je     12b <half_to_float+0x11b>
 1cd:	e9 6e fe ff ff       	jmp    40 <half_to_float+0x30>
 1d2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
 1d9:	1f 84 00 00 00 00 00 

00000000000001e0 <bf16_to_float>:
 1e0:	85 d2                	test   %edx,%edx
 1e2:	0f 8e cf 00 00 00    	jle    2b7 <bf16_to_float+0xd7>
 1e8:	48 63 c2             	movslq %edx,%rax
 1eb:	83 fa 04             	cmp    $0x4,%edx
 1ee:	72 1e                	jb     20e <bf16_to_float+0x2e>
 1f0:	4c 8d 04 46          	lea    (%rsi,%rax,2),%r8
 1f4:	48 8d 0c 87          	lea    (%rdi,%rax,4),%rcx
 1f8:	4c 39 c7             	cmp    %r8,%rdi
 1fb:	41 0f 92 c0          	setb   %r8b
 1ff:	48 39 ce             	cmp    %rcx,%rsi
 202:	0f 92 c1             	setb   %cl
 205:	41 84 c8             	test   %cl,%r8b
 208:	0f 84 ac 00 00 00    	je     2ba <bf16_to_float+0xda>
 20e:	31 c9                	xor    %ecx,%ecx
 210:	29 ca                	sub    %ecx,%edx
 212:	49 89 c8             	mov    %rcx,%r8
 215:	83 e2 07             	and    $0x7,%edx
 218:	74 1b                	je     235 <bf16_to_float+0x55>
 21a:	49 89 c8             	mov    %rcx,%r8
 21d:	0f 1f 00             	nopl   (%rax)
 220:	46 0f b7 0c 46       	movzwl (%rsi,%r8,2),%r9d
 225:	41 c1 e1 10          	shl    $0x10,%r9d
 229:	46 89 0c 87          	mov    %r9d,(%rdi,%r8,4)
 22d:	49 ff c0             	inc    %r8
 230:	48 ff ca             	dec    %rdx
 233:	75 eb                	jne    220 <bf16_to_float+0x40>
 235:	48 29 c1             	sub    %rax,%rcx
 238:	48 83 f9 f8          	cmp    $0xfffffffffffffff8,%rcx
 23c:	77 79                	ja     2b7 <bf16_to_float+0xd7>
 23e:	66 90                	xchg   %ax,%ax
 240:	42 0f b7 0c 46       	movzwl (%rsi,%r8,2),%ecx
 245:	c1 e1 10             	shl    $0x10,%ecx
 248:	42 89 0c 87          	mov    %ecx,(%rdi,%r8,4)
 24c:	42 0f b7 4c 46 02    	movzwl 0x2(%rsi,%r8,2),%ecx
 252:	c1 e1 10             	shl    $0x10,%ecx
 255:	42 89 4c 87 04       	mov    %ecx,0x4(%rdi,%r8,4)
 25a:	42 0f b7 4c 46 04    	movzwl 0x4(%rsi,%r8,2),%ecx
 260:	c1 e1 10             	shl    $0x10,%ecx
 263:	42 89 4c 87 08       	mov    %ecx,0x8(%rdi,%r8,4)
 268:	42 0f b7 4c 46 06    	movzwl 0x6(%rsi,%r8,2),%ecx
 26e:	c1 e1 10             	shl    $0x10,%ecx
 271:	42 89 4c 87 0c       	mov    %ecx,0xc(%rdi,%r8,4)
 276:	42 0f b7 4c 46 08    	movzwl 0x8(%rsi,%r8,2),%ecx
 27c:	c1 e1 10             	shl    $0x10,%ecx
 27f:	42 89 4c 87 10       	mov    %ecx,0x10(%rdi,%r8,4)
 284:	42 0f b7 4c 46 0a    	movzwl 0xa(%rsi,%r8,2),%ecx
 28a:	c1 e1 10             	shl    $0x10,%ecx
 28d:	42 89 4c 87 14       	mov    %ecx,0x14(%rdi,%r8,4)
 292:	42 0f b7 4c 46 0c    	movzwl 0xc(%rsi,%r8,2),%ecx
 298:	c1 e1 10             	shl    $0x10,%ecx
 29b:	42 89 4c 87 18       	mov    %ecx,0x18(%rdi,%r8,4)
 2a0:	42 0f b7 4c 46 0e    	movzwl 0xe(%rsi,%r8,2),%ecx
 2a6:	c1 e1 10             	shl    $0x10,%ecx
 2a9:	42 89 4c 87 1c       	mov    %ecx,0x1c(%rdi,%r8,4)
 2ae:	49 83 c0 08          	add    $0x8,%r8
 2b2:	4c 39 c0             	cmp    %r8,%rax
 2b5:	75 89                	jne    240 <bf16_to_float+0x60>
 2b7:	31 c0                	xor    %eax,%eax
 2b9:	c3                   	ret    
 2ba:	83 fa 10             	cmp    $0x10,%edx
 2bd:	73 04                	jae    2c3 <bf16_to_float+0xe3>
 2bf:	31 c9                	xor    %ecx,%ecx
 2c1:	eb 7d                	jmp    340 <bf16_to_float+0x160>
 2c3:	41 89 d0             	mov    %edx,%r8d
 2c6:	41 c1 e8 04          	shr    $0x4,%r8d
 2ca:	89 d1                	mov    %edx,%ecx
 2cc:	81 e1 f0 ff ff 7f    	and    $0x7ffffff0,%ecx
 2d2:	45 31 c9             	xor    %r9d,%r9d
 2d5:	41 c1 e0 05          	shl    $0x5,%r8d
 2d9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
 2e0:	c4 a2 79 33 04 0e    	vpmovzxwd (%rsi,%r9,1),%xmm0
 2e6:	c4 a2 79 33 4c 0e 08 	vpmovzxwd 0x8(%rsi,%r9,1),%xmm1
 2ed:	c4 a2 79 33 54 0e 10 	vpmovzxwd 0x10(%rsi,%r9,1),%xmm2
 2f4:	c4 a2 79 33 5c 0e 18 	vpmovzxwd 0x18(%rsi,%r9,1),%xmm3
 2fb:	c5 f9 72 f0 10       	vpslld $0x10,%xmm0,%xmm0
 300:	c5 f1 72 f1 10       	vpslld $0x10,%xmm1,%xmm1
 305:	c5 e9 72 f2 10       	vpslld $0x10,%xmm2,%xmm2
 30a:	c5 e1 72 f3 10       	vpslld $0x10,%xmm3,%xmm3
 30f:	c4 a1 7a 7f 04 4f    	vmovdqu %xmm0,(%rdi,%r9,2)
 315:	c4 a1 7a 7f 4c 4f 10 	vmovdqu %xmm1,0x10(%rdi,%r9,2)
 31c:	c4 a1 7a 7f 54 4f 20 	vmovdqu %xmm2,0x20(%rdi,%r9,2)
 323:	c4 a1 7a 7f 5c 4f 30 	vmovdqu %xmm3,0x30(%rdi,%r9,2)
 32a:	49 83 c1 20          	add    $0x20,%r9
 32e:	4d 39 c8             	cmp    %r9,%r8
 331:	75 ad                	jne    2e0 <bf16_to_float+0x100>
 333:	39 d1                	cmp    %edx,%ecx
 335:	74 80                	je     2b7 <bf16_to_float+0xd7>
 337:	f6 c2 0c             	test   $0xc,%dl
 33a:	0f 84 d0 fe ff ff    	je     210 <bf16_to_float+0x30>
 340:	49 89 c8             	mov    %rcx,%r8
 343:	89 d1                	mov    %edx,%ecx
 345:	81 e1 fc ff ff 7f    	and    $0x7ffffffc,%ecx
 34b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
 350:	c4 a2 79 33 04 46    	vpmovzxwd (%rsi,%r8,2),%xmm0
 356:	c5 f9 72 f0 10       	vpslld $0x10,%xmm0,%xmm0
 35b:	c4 a1 7a 7f 04 87    	vmovdqu %xmm0,(%rdi,%r8,4)
 361:	49 83 c0 04          	add    $0x4,%r8
 365:	4c 39 c1             	cmp    %r8,%rcx
 368:	75 e6                	jne    350 <bf16_to_float+0x170>
 36a:	39 d1                	cmp    %edx,%ecx
 36c:	0f 84 45 ff ff ff    	je     2b7 <bf16_to_float+0xd7>
 372:	e9 99 fe ff ff       	jmp    210 <bf16_to_float+0x30>
 377:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
 37e:	00 00 

0000000000000380 <_mlir_vector_add>:
 380:	48 8b 07             	mov    (%rdi),%rax
 383:	48 8b 4f 08          	mov    0x8(%rdi),%rcx
 387:	c5 f8 28 00          	vmovaps (%rax),%xmm0
 38b:	48 8b 47 10          	mov    0x10(%rdi),%rax
 38f:	c5 f8 58 01          	vaddps (%rcx),%xmm0,%xmm0
 393:	c5 f8 29 00          	vmovaps %xmm0,(%rax)
 397:	c3                   	ret    
 398:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
 39f:	00 

00000000000003a0 <_mlir_half_to_float>:
 3a0:	48 8b 47 10          	mov    0x10(%rdi),%rax
 3a4:	48 63 00             	movslq (%rax),%rax
 3a7:	48 85 c0             	test   %rax,%rax
 3aa:	0f 8e 2b 01 00 00    	jle    4db <_mlir_half_to_float+0x13b>
 3b0:	48 8b 0f             	mov    (%rdi),%rcx
 3b3:	48 8b 57 08          	mov    0x8(%rdi),%rdx
 3b7:	48 8b 09             	mov    (%rcx),%rcx
 3ba:	48 8b 12             	mov    (%rdx),%rdx
 3bd:	83 f8 08             	cmp    $0x8,%eax
 3c0:	72 1f                	jb     3e1 <_mlir_half_to_float+0x41>
 3c2:	4c 8d 04 42          	lea    (%rdx,%rax,2),%r8
 3c6:	48 8d 34 81          	lea    (%rcx,%rax,4),%rsi
 3ca:	4c 39 c1             	cmp    %r8,%rcx
 3cd:	41 0f 92 c0          	setb   %r8b
 3d1:	48 39 f2             	cmp    %rsi,%rdx
 3d4:	40 0f 92 c6          	setb   %sil
 3d8:	41 84 f0             	test   %sil,%r8b
 3db:	0f 84 08 01 00 00    	je     4e9 <_mlir_half_to_float+0x149>
 3e1:	31 f6                	xor    %esi,%esi
 3e3:	49 89 f0             	mov    %rsi,%r8
 3e6:	a8 07                	test   $0x7,%al
 3e8:	74 30                	je     41a <_mlir_half_to_float+0x7a>
 3ea:	41 89 c1             	mov    %eax,%r9d
 3ed:	41 83 e1 07          	and    $0x7,%r9d
 3f1:	49 89 f0             	mov    %rsi,%r8
 3f4:	66 66 66 2e 0f 1f 84 	data16 data16 cs nopw 0x0(%rax,%rax,1)
 3fb:	00 00 00 00 00 
 400:	c4 a1 79 c4 04 42 00 	vpinsrw $0x0,(%rdx,%r8,2),%xmm0,%xmm0
 407:	c4 e2 79 13 c0       	vcvtph2ps %xmm0,%xmm0
 40c:	c4 a1 79 7e 04 81    	vmovd  %xmm0,(%rcx,%r8,4)
 412:	49 ff c0             	inc    %r8
 415:	49 ff c9             	dec    %r9
 418:	75 e6                	jne    400 <_mlir_half_to_float+0x60>
 41a:	48 29 c6             	sub    %rax,%rsi
 41d:	48 83 fe f8          	cmp    $0xfffffffffffffff8,%rsi
 421:	0f 87 b4 00 00 00    	ja     4db <_mlir_half_to_float+0x13b>
 427:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
 42e:	00 00 
 430:	c4 a1 79 c4 04 42 00 	vpinsrw $0x0,(%rdx,%r8,2),%xmm0,%xmm0
 437:	c4 e2 79 13 c0       	vcvtph2ps %xmm0,%xmm0
 43c:	c4 a1 79 7e 04 81    	vmovd  %xmm0,(%rcx,%r8,4)
 442:	c4 a1 79 c4 44 42 02 	vpinsrw $0x0,0x2(%rdx,%r8,2),%xmm0,%xmm0
 449:	00 
 44a:	c4 e2 79 13 c0       	vcvtph2ps %xmm0,%xmm0
 44f:	c4 a1 79 7e 44 81 04 	vmovd  %xmm0,0x4(%rcx,%r8,4)
 456:	c4 a1 79 c4 44 42 04 	vpinsrw $0x0,0x4(%rdx,%r8,2),%xmm0,%xmm0
 45d:	00 
 45e:	c4 e2 79 13 c0       	vcvtph2ps %xmm0,%xmm0
 463:	c4 a1 79 7e 44 81 08 	vmovd  %xmm0,0x8(%rcx,%r8,4)
 46a:	c4 a1 79 c4 44 42 06 	vpinsrw $0x0,0x6(%rdx,%r8,2),%xmm0,%xmm0
 471:	00 
 472:	c4 e2 79 13 c0       	vcvtph2ps %xmm0,%xmm0
 477:	c4 a1 79 7e 44 81 0c 	vmovd  %xmm0,0xc(%rcx,%r8,4)
 47e:	c4 a1 79 c4 44 42 08 	vpinsrw $0x0,0x8(%rdx,%r8,2),%xmm0,%xmm0
 485:	00 
 486:	c4 e2 79 13 c0       	vcvtph2ps %xmm0,%xmm0
 48b:	c4 a1 79 7e 44 81 10 	vmovd  %xmm0,0x10(%rcx,%r8,4)
 492:	c4 a1 79 c4 44 42 0a 	vpinsrw $0x0,0xa(%rdx,%r8,2),%xmm0,%xmm0
 499:	00 
 49a:	c4 e2 79 13 c0       	vcvtph2ps %xmm0,%xmm0
 49f:	c4 a1 79 7e 44 81 14 	vmovd  %xmm0,0x14(%rcx,%r8,4)
 4a6:	c4 a1 79 c4 44 42 0c 	vpinsrw $0x0,0xc(%rdx,%r8,2),%xmm0,%xmm0
 4ad:	00 
 4ae:	c4 e2 79 13 c0       	vcvtph2ps %xmm0,%xmm0
 4b3:	c4 a1 79 7e 44 81 18 	vmovd  %xmm0,0x18(%rcx,%r8,4)
 4ba:	c4 a1 79 c4 44 42 0e 	vpinsrw $0x0,0xe(%rdx,%r8,2),%xmm0,%xmm0
 4c1:	00 
 4c2:	c4 e2 79 13 c0       	vcvtph2ps %xmm0,%xmm0
 4c7:	c4 a1 79 7e 44 81 1c 	vmovd  %xmm0,0x1c(%rcx,%r8,4)
 4ce:	49 83 c0 08          	add    $0x8,%r8
 4d2:	4c 39 c0             	cmp    %r8,%rax
 4d5:	0f 85 55 ff ff ff    	jne    430 <_mlir_half_to_float+0x90>
 4db:	48 8b 47 18          	mov    0x18(%rdi),%rax
 4df:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
 4e5:	c5 f8 77             	vzeroupper 
 4e8:	c3                   	ret    
 4e9:	83 f8 40             	cmp    $0x40,%eax
 4ec:	73 04                	jae    4f2 <_mlir_half_to_float+0x152>
 4ee:	31 f6                	xor    %esi,%esi
 4f0:	eb 75                	jmp    567 <_mlir_half_to_float+0x1c7>
 4f2:	41 89 c0             	mov    %eax,%r8d
 4f5:	41 c1 e8 06          	shr    $0x6,%r8d
 4f9:	89 c6                	mov    %eax,%esi
 4fb:	81 e6 c0 ff ff 7f    	and    $0x7fffffc0,%esi
 501:	45 31 c9             	xor    %r9d,%r9d
 504:	41 c1 e0 07          	shl    $0x7,%r8d
 508:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
 50f:	00 
 510:	62 b2 7d 48 13 04 0a 	vcvtph2ps (%rdx,%r9,1),%zmm0
 517:	62 b2 7d 48 13 4c 0a 	vcvtph2ps 0x20(%rdx,%r9,1),%zmm1
 51e:	01 
 51f:	62 b2 7d 48 13 54 0a 	vcvtph2ps 0x40(%rdx,%r9,1),%zmm2
 526:	02 
 527:	62 b2 7d 48 13 5c 0a 	vcvtph2ps 0x60(%rdx,%r9,1),%zmm3
 52e:	03 
 52f:	62 b1 fe 48 7f 04 49 	vmovdqu64 %zmm0,(%rcx,%r9,2)
 536:	62 b1 7c 48 11 4c 49 	vmovups %zmm1,0x40(%rcx,%r9,2)
 53d:	01 
 53e:	62 b1 7c 48 11 54 49 	vmovups %zmm2,0x80(%rcx,%r9,2)
 545:	02 
 546:	62 b1 7c 48 11 5c 49 	vmovups %zmm3,0xc0(%rcx,%r9,2)
 54d:	03 
 54e:	49 83 e9 80          	sub    $0xffffffffffffff80,%r9
 552:	4d 39 c8             	cmp    %r9,%r8
 555:	75 b9                	jne    510 <_mlir_half_to_float+0x170>
 557:	39 c6                	cmp    %eax,%esi
 559:	0f 84 7c ff ff ff    	je     4db <_mlir_half_to_float+0x13b>
 55f:	a8 38                	test   $0x38,%al
 561:	0f 84 7c fe ff ff    	je     3e3 <_mlir_half_to_float+0x43>
 567:	49 89 f0             	mov    %rsi,%r8
 56a:	89 c6                	mov    %eax,%esi
 56c:	81 e6 f8 ff ff 7f    	and    $0x7ffffff8,%esi
 572:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
 579:	1f 84 00 00 00 00 00 
 580:	c4 a2 7d 13 04 42    	vcvtph2ps (%rdx,%r8,2),%ymm0
 586:	c4 a1 7e 7f 04 81    	vmovdqu %ymm0,(%rcx,%r8,4)
 58c:	49 83 c0 08          	add    $0x8,%r8
 590:	4c 39 c6             	cmp    %r8,%rsi
 593:	75 eb                	jne    580 <_mlir_half_to_float+0x1e0>
 595:	39 c6                	cmp    %eax,%esi
 597:	0f 84 3e ff ff ff    	je     4db <_mlir_half_to_float+0x13b>
 59d:	e9 41 fe ff ff       	jmp    3e3 <_mlir_half_to_float+0x43>
 5a2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
 5a9:	1f 84 00 00 00 00 00 

00000000000005b0 <_mlir_bf16_to_float>:
 5b0:	48 8b 47 10          	mov    0x10(%rdi),%rax
 5b4:	48 63 00             	movslq (%rax),%rax
 5b7:	48 85 c0             	test   %rax,%rax
 5ba:	0f 8e e7 00 00 00    	jle    6a7 <_mlir_bf16_to_float+0xf7>
 5c0:	48 8b 0f             	mov    (%rdi),%rcx
 5c3:	48 8b 57 08          	mov    0x8(%rdi),%rdx
 5c7:	48 8b 09             	mov    (%rcx),%rcx
 5ca:	48 8b 12             	mov    (%rdx),%rdx
 5cd:	83 f8 04             	cmp    $0x4,%eax
 5d0:	72 1f                	jb     5f1 <_mlir_bf16_to_float+0x41>
 5d2:	4c 8d 04 42          	lea    (%rdx,%rax,2),%r8
 5d6:	48 8d 34 81          	lea    (%rcx,%rax,4),%rsi
 5da:	4c 39 c1             	cmp    %r8,%rcx
 5dd:	41 0f 92 c0          	setb   %r8b
 5e1:	48 39 f2             	cmp    %rsi,%rdx
 5e4:	40 0f 92 c6          	setb   %sil
 5e8:	41 84 f0             	test   %sil,%r8b
 5eb:	0f 84 c1 00 00 00    	je     6b2 <_mlir_bf16_to_float+0x102>
 5f1:	31 f6                	xor    %esi,%esi
 5f3:	41 89 c1             	mov    %eax,%r9d
 5f6:	41 29 f1             	sub    %esi,%r9d
 5f9:	49 89 f0             	mov    %rsi,%r8
 5fc:	41 83 e1 07          	and    $0x7,%r9d
 600:	74 23                	je     625 <_mlir_bf16_to_float+0x75>
 602:	49 89 f0             	mov    %rsi,%r8
 605:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
 60c:	00 00 00 00 
 610:	46 0f b7 14 42       	movzwl (%rdx,%r8,2),%r10d
 615:	41 c1 e2 10          	shl    $0x10,%r10d
 619:	46 89 14 81          	mov    %r10d,(%rcx,%r8,4)
 61d:	49 ff c0             	inc    %r8
 620:	49 ff c9             	dec    %r9
 623:	75 eb                	jne    610 <_mlir_bf16_to_float+0x60>
 625:	48 29 c6             	sub    %rax,%rsi
 628:	48 83 fe f8          	cmp    $0xfffffffffffffff8,%rsi
 62c:	77 79                	ja     6a7 <_mlir_bf16_to_float+0xf7>
 62e:	66 90                	xchg   %ax,%ax
 630:	42 0f b7 34 42       	movzwl (%rdx,%r8,2),%esi
 635:	c1 e6 10             	shl    $0x10,%esi
 638:	42 89 34 81          	mov    %esi,(%rcx,%r8,4)
 63c:	42 0f b7 74 42 02    	movzwl 0x2(%rdx,%r8,2),%esi
 642:	c1 e6 10             	shl    $0x10,%esi
 645:	42 89 74 81 04       	mov    %esi,0x4(%rcx,%r8,4)
 64a:	42 0f b7 74 42 04    	movzwl 0x4(%rdx,%r8,2),%esi
 650:	c1 e6 10             	shl    $0x10,%esi
 653:	42 89 74 81 08       	mov    %esi,0x8(%rcx,%r8,4)
 658:	42 0f b7 74 42 06    	movzwl 0x6(%rdx,%r8,2),%esi
 65e:	c1 e6 10             	shl    $0x10,%esi
 661:	42 89 74 81 0c       	mov    %esi,0xc(%rcx,%r8,4)
 666:	42 0f b7 74 42 08    	movzwl 0x8(%rdx,%r8,2),%esi
 66c:	c1 e6 10             	shl    $0x10,%esi
 66f:	42 89 74 81 10       	mov    %esi,0x10(%rcx,%r8,4)
 674:	42 0f b7 74 42 0a    	movzwl 0xa(%rdx,%r8,2),%esi
 67a:	c1 e6 10             	shl    $0x10,%esi
 67d:	42 89 74 81 14       	mov    %esi,0x14(%rcx,%r8,4)
 682:	42 0f b7 74 42 0c    	movzwl 0xc(%rdx,%r8,2),%esi
 688:	c1 e6 10             	shl    $0x10,%esi
 68b:	42 89 74 81 18       	mov    %esi,0x18(%rcx,%r8,4)
 690:	42 0f b7 74 42 0e    	movzwl 0xe(%rdx,%r8,2),%esi
 696:	c1 e6 10             	shl    $0x10,%esi
 699:	42 89 74 81 1c       	mov    %esi,0x1c(%rcx,%r8,4)
 69e:	49 83 c0 08          	add    $0x8,%r8
 6a2:	4c 39 c0             	cmp    %r8,%rax
 6a5:	75 89                	jne    630 <_mlir_bf16_to_float+0x80>
 6a7:	48 8b 47 18          	mov    0x18(%rdi),%rax
 6ab:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
 6b1:	c3                   	ret    
 6b2:	83 f8 10             	cmp    $0x10,%eax
 6b5:	73 07                	jae    6be <_mlir_bf16_to_float+0x10e>
 6b7:	31 f6                	xor    %esi,%esi
 6b9:	e9 85 00 00 00       	jmp    743 <_mlir_bf16_to_float+0x193>
 6be:	41 89 c0             	mov    %eax,%r8d
 6c1:	41 c1 e8 04          	shr    $0x4,%r8d
 6c5:	89 c6                	mov    %eax,%esi
 6c7:	81 e6 f0 ff ff 7f    	and    $0x7ffffff0,%esi
 6cd:	45 31 c9             	xor    %r9d,%r9d
 6d0:	41 c1 e0 05          	shl    $0x5,%r8d
 6d4:	66 66 66 2e 0f 1f 84 	data16 data16 cs nopw 0x0(%rax,%rax,1)
 6db:	00 00 00 00 00 
 6e0:	c4 a2 79 33 04 0a    	vpmovzxwd (%rdx,%r9,1),%xmm0
 6e6:	c4 a2 79 33 4c 0a 08 	vpmovzxwd 0x8(%rdx,%r9,1),%xmm1
 6ed:	c4 a2 79 33 54 0a 10 	vpmovzxwd 0x10(%rdx,%r9,1),%xmm2
 6f4:	c4 a2 79 33 5c 0a 18 	vpmovzxwd 0x18(%rdx,%r9,1),%xmm3
 6fb:	c5 f9 72 f0 10       	vpslld $0x10,%xmm0,%xmm0
 700:	c5 f1 72 f1 10       	vpslld $0x10,%xmm1,%xmm1
 705:	c5 e9 72 f2 10       	vpslld $0x10,%xmm2,%xmm2
 70a:	c5 e1 72 f3 10       	vpslld $0x10,%xmm3,%xmm3
 70f:	c4 a1 7a 7f 04 49    	vmovdqu %xmm0,(%rcx,%r9,2)
 715:	c4 a1 7a 7f 4c 49 10 	vmovdqu %xmm1,0x10(%rcx,%r9,2)
 71c:	c4 a1 7a 7f 54 49 20 	vmovdqu %xmm2,0x20(%rcx,%r9,2)
 723:	c4 a1 7a 7f 5c 49 30 	vmovdqu %xmm3,0x30(%rcx,%r9,2)
 72a:	49 83 c1 20          	add    $0x20,%r9
 72e:	4d 39 c8             	cmp    %r9,%r8
 731:	75 ad                	jne    6e0 <_mlir_bf16_to_float+0x130>
 733:	39 c6                	cmp    %eax,%esi
 735:	0f 84 6c ff ff ff    	je     6a7 <_mlir_bf16_to_float+0xf7>
 73b:	a8 0c                	test   $0xc,%al
 73d:	0f 84 b0 fe ff ff    	je     5f3 <_mlir_bf16_to_float+0x43>
 743:	49 89 f0             	mov    %rsi,%r8
 746:	89 c6                	mov    %eax,%esi
 748:	81 e6 fc ff ff 7f    	and    $0x7ffffffc,%esi
 74e:	66 90                	xchg   %ax,%ax
 750:	c4 a2 79 33 04 42    	vpmovzxwd (%rdx,%r8,2),%xmm0
 756:	c5 f9 72 f0 10       	vpslld $0x10,%xmm0,%xmm0
 75b:	c4 a1 7a 7f 04 81    	vmovdqu %xmm0,(%rcx,%r8,4)
 761:	49 83 c0 04          	add    $0x4,%r8
 765:	4c 39 c6             	cmp    %r8,%rsi
 768:	75 e6                	jne    750 <_mlir_bf16_to_float+0x1a0>
 76a:	39 c6                	cmp    %eax,%esi
 76c:	0f 84 35 ff ff ff    	je     6a7 <_mlir_bf16_to_float+0xf7>
 772:	e9 7c fe ff ff       	jmp    5f3 <_mlir_bf16_to_float+0x43>
