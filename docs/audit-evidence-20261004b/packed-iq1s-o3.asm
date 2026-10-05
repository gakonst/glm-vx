
kernels/build-audit-tiled-o3/libglm_vx.so:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .text:

Disassembly of section .ltext:

000000000000df40 <glm_vx_packed_batch_v2_iq1_s>:
    df40:	55                   	push   rbp
    df41:	41 57                	push   r15
    df43:	41 56                	push   r14
    df45:	41 55                	push   r13
    df47:	41 54                	push   r12
    df49:	53                   	push   rbx
    df4a:	48 81 ec c8 01 00 00 	sub    rsp,0x1c8
    df51:	8b 9c 24 00 02 00 00 	mov    ebx,DWORD PTR [rsp+0x200]
    df58:	48 8d 05 f9 ff ff ff 	lea    rax,[rip+0xfffffffffffffff9]        # df58 <glm_vx_packed_batch_v2_iq1_s+0x18>
    df5f:	49 ba a8 b0 01 00 00 	movabs r10,0x1b0a8
    df66:	00 00 00 
    df69:	44 8b 9c 24 08 02 00 	mov    r11d,DWORD PTR [rsp+0x208]
    df70:	00 
    df71:	48 89 0c 24          	mov    QWORD PTR [rsp],rcx
    df75:	48 89 7c 24 f8       	mov    QWORD PTR [rsp-0x8],rdi
    df7a:	49 01 c2             	add    r10,rax
    df7d:	45 85 c9             	test   r9d,r9d
    df80:	0f 98 c0             	sets   al
    df83:	85 db                	test   ebx,ebx
    df85:	0f 9e c1             	setle  cl
    df88:	84 db                	test   bl,bl
    df8a:	40 0f 95 c7          	setne  dil
    df8e:	40 08 cf             	or     dil,cl
    df91:	41 83 fb 41          	cmp    r11d,0x41
    df95:	0f 93 c1             	setae  cl
    df98:	08 c1                	or     cl,al
    df9a:	b8 ff ff ff ff       	mov    eax,0xffffffff
    df9f:	40 08 f9             	or     cl,dil
    dfa2:	f6 c1 01             	test   cl,0x1
    dfa5:	0f 85 7a 1e 00 00    	jne    fe25 <glm_vx_packed_batch_v2_iq1_s+0x1ee5>
    dfab:	89 d8                	mov    eax,ebx
    dfad:	c1 e8 08             	shr    eax,0x8
    dfb0:	45 31 f6             	xor    r14d,r14d
    dfb3:	4c 89 4c 24 e8       	mov    QWORD PTR [rsp-0x18],r9
    dfb8:	4c 89 54 24 08       	mov    QWORD PTR [rsp+0x8],r10
    dfbd:	89 44 24 84          	mov    DWORD PTR [rsp-0x7c],eax
    dfc1:	41 83 fb 40          	cmp    r11d,0x40
    dfc5:	0f 82 14 0a 00 00    	jb     e9df <glm_vx_packed_batch_v2_iq1_s+0xa9f>
    dfcb:	45 85 c9             	test   r9d,r9d
    dfce:	0f 84 4f 1e 00 00    	je     fe23 <glm_vx_packed_batch_v2_iq1_s+0x1ee3>
    dfd4:	43 8d 04 09          	lea    eax,[r9+r9*1]
    dfd8:	43 8d 0c 49          	lea    ecx,[r9+r9*2]
    dfdc:	47 8d 1c 89          	lea    r11d,[r9+r9*4]
    dfe0:	43 8d 2c c9          	lea    ebp,[r9+r9*8]
    dfe4:	42 8d 3c 8d 00 00 00 	lea    edi,[r9*4+0x0]
    dfeb:	00 
    dfec:	8d 1c 40             	lea    ebx,[rax+rax*2]
    dfef:	48 89 44 24 90       	mov    QWORD PTR [rsp-0x70],rax
    dff4:	48 89 4c 24 98       	mov    QWORD PTR [rsp-0x68],rcx
    dff9:	4c 89 5c 24 88       	mov    QWORD PTR [rsp-0x78],r11
    dffe:	48 89 7c 24 b0       	mov    QWORD PTR [rsp-0x50],rdi
    e003:	48 89 ac 24 98 01 00 	mov    QWORD PTR [rsp+0x198],rbp
    e00a:	00 
    e00b:	48 89 5c 24 a8       	mov    QWORD PTR [rsp-0x58],rbx
    e010:	42 8d 1c cd 00 00 00 	lea    ebx,[r9*8+0x0]
    e017:	00 
    e018:	41 89 de             	mov    r14d,ebx
    e01b:	45 29 ce             	sub    r14d,r9d
    e01e:	48 89 5c 24 e0       	mov    QWORD PTR [rsp-0x20],rbx
    e023:	4c 89 74 24 a0       	mov    QWORD PTR [rsp-0x60],r14
    e028:	45 89 ce             	mov    r14d,r9d
    e02b:	41 c1 e6 04          	shl    r14d,0x4
    e02f:	45 89 f7             	mov    r15d,r14d
    e032:	41 29 c7             	sub    r15d,eax
    e035:	4c 89 74 24 f0       	mov    QWORD PTR [rsp-0x10],r14
    e03a:	4c 89 7c 24 10       	mov    QWORD PTR [rsp+0x10],r15
    e03f:	44 8d 3c cd 00 00 00 	lea    r15d,[rcx*8+0x0]
    e046:	00 
    e047:	41 8d 0c 89          	lea    ecx,[r9+rcx*4]
    e04b:	45 29 cf             	sub    r15d,r9d
    e04e:	48 89 8c 24 78 01 00 	mov    QWORD PTR [rsp+0x178],rcx
    e055:	00 
    e056:	43 8d 0c 5b          	lea    ecx,[r11+r11*2]
    e05a:	4c 89 7c 24 18       	mov    QWORD PTR [rsp+0x18],r15
    e05f:	45 89 cf             	mov    r15d,r9d
    e062:	41 c1 e7 05          	shl    r15d,0x5
    e066:	45 89 fc             	mov    r12d,r15d
    e069:	41 29 c4             	sub    r12d,eax
    e06c:	48 89 8c 24 70 01 00 	mov    QWORD PTR [rsp+0x170],rcx
    e073:	00 
    e074:	43 8d 0c 31          	lea    ecx,[r9+r14*1]
    e078:	4c 89 bc 24 c0 01 00 	mov    QWORD PTR [rsp+0x1c0],r15
    e07f:	00 
    e080:	4c 89 a4 24 b8 01 00 	mov    QWORD PTR [rsp+0x1b8],r12
    e087:	00 
    e088:	45 89 fc             	mov    r12d,r15d
    e08b:	45 29 cc             	sub    r12d,r9d
    e08e:	48 89 8c 24 68 01 00 	mov    QWORD PTR [rsp+0x168],rcx
    e095:	00 
    e096:	8d 0c 5b             	lea    ecx,[rbx+rbx*2]
    e099:	4c 89 a4 24 b0 01 00 	mov    QWORD PTR [rsp+0x1b0],r12
    e0a0:	00 
    e0a1:	45 89 cc             	mov    r12d,r9d
    e0a4:	41 c1 e4 06          	shl    r12d,0x6
    e0a8:	45 89 e5             	mov    r13d,r12d
    e0ab:	41 29 c5             	sub    r13d,eax
    e0ae:	48 89 8c 24 40 01 00 	mov    QWORD PTR [rsp+0x140],rcx
    e0b5:	00 
    e0b6:	43 8d 0c 9b          	lea    ecx,[r11+r11*4]
    e0ba:	45 29 cc             	sub    r12d,r9d
    e0bd:	4c 89 ac 24 a0 01 00 	mov    QWORD PTR [rsp+0x1a0],r13
    e0c4:	00 
    e0c5:	44 8d 2c 80          	lea    r13d,[rax+rax*4]
    e0c9:	8d 04 c0             	lea    eax,[rax+rax*8]
    e0cc:	4c 89 a4 24 a8 01 00 	mov    QWORD PTR [rsp+0x1a8],r12
    e0d3:	00 
    e0d4:	47 8d 24 59          	lea    r12d,[r9+r11*2]
    e0d8:	48 89 8c 24 38 01 00 	mov    QWORD PTR [rsp+0x138],rcx
    e0df:	00 
    e0e0:	48 89 84 24 60 01 00 	mov    QWORD PTR [rsp+0x160],rax
    e0e7:	00 
    e0e8:	41 8d 04 69          	lea    eax,[r9+rbp*2]
    e0ec:	4c 89 ac 24 90 01 00 	mov    QWORD PTR [rsp+0x190],r13
    e0f3:	00 
    e0f4:	49 bd 40 e4 ff ff ff 	movabs r13,0xffffffffffffe440
    e0fb:	ff ff ff 
    e0fe:	4c 89 a4 24 88 01 00 	mov    QWORD PTR [rsp+0x188],r12
    e105:	00 
    e106:	44 8d 24 7f          	lea    r12d,[rdi+rdi*2]
    e10a:	4d 01 d5             	add    r13,r10
    e10d:	41 ba 03 02 00 00    	mov    r10d,0x203
    e113:	48 89 84 24 58 01 00 	mov    QWORD PTR [rsp+0x158],rax
    e11a:	00 
    e11b:	8d 04 bf             	lea    eax,[rdi+rdi*4]
    e11e:	4c 89 a4 24 80 01 00 	mov    QWORD PTR [rsp+0x180],r12
    e125:	00 
    e126:	48 89 84 24 50 01 00 	mov    QWORD PTR [rsp+0x150],rax
    e12d:	00 
    e12e:	43 8d 04 99          	lea    eax,[r9+r11*4]
    e132:	48 89 84 24 48 01 00 	mov    QWORD PTR [rsp+0x148],rax
    e139:	00 
    e13a:	44 01 c8             	add    eax,r9d
    e13d:	48 89 84 24 30 01 00 	mov    QWORD PTR [rsp+0x130],rax
    e144:	00 
    e145:	42 8d 04 09          	lea    eax,[rcx+r9*1]
    e149:	48 89 84 24 28 01 00 	mov    QWORD PTR [rsp+0x128],rax
    e150:	00 
    e151:	8d 44 6d 00          	lea    eax,[rbp+rbp*2+0x0]
    e155:	48 89 84 24 20 01 00 	mov    QWORD PTR [rsp+0x120],rax
    e15c:	00 
    e15d:	44 01 c8             	add    eax,r9d
    e160:	48 89 84 24 18 01 00 	mov    QWORD PTR [rsp+0x118],rax
    e167:	00 
    e168:	44 01 c8             	add    eax,r9d
    e16b:	48 89 84 24 10 01 00 	mov    QWORD PTR [rsp+0x110],rax
    e172:	00 
    e173:	43 8d 04 39          	lea    eax,[r9+r15*1]
    e177:	48 89 84 24 08 01 00 	mov    QWORD PTR [rsp+0x108],rax
    e17e:	00 
    e17f:	43 8d 04 4f          	lea    eax,[r15+r9*2]
    e183:	48 89 84 24 00 01 00 	mov    QWORD PTR [rsp+0x100],rax
    e18a:	00 
    e18b:	41 6b c1 23          	imul   eax,r9d,0x23
    e18f:	48 89 84 24 f8 00 00 	mov    QWORD PTR [rsp+0xf8],rax
    e196:	00 
    e197:	8d 04 ff             	lea    eax,[rdi+rdi*8]
    e19a:	31 ff                	xor    edi,edi
    e19c:	48 89 84 24 f0 00 00 	mov    QWORD PTR [rsp+0xf0],rax
    e1a3:	00 
    e1a4:	41 8d 04 a9          	lea    eax,[r9+rbp*4]
    e1a8:	48 89 84 24 e8 00 00 	mov    QWORD PTR [rsp+0xe8],rax
    e1af:	00 
    e1b0:	41 6b c1 26          	imul   eax,r9d,0x26
    e1b4:	48 89 84 24 e0 00 00 	mov    QWORD PTR [rsp+0xe0],rax
    e1bb:	00 
    e1bc:	41 6b c1 27          	imul   eax,r9d,0x27
    e1c0:	48 89 84 24 d8 00 00 	mov    QWORD PTR [rsp+0xd8],rax
    e1c7:	00 
    e1c8:	8d 04 9b             	lea    eax,[rbx+rbx*4]
    e1cb:	48 89 84 24 d0 00 00 	mov    QWORD PTR [rsp+0xd0],rax
    e1d2:	00 
    e1d3:	43 8d 04 d9          	lea    eax,[r9+r11*8]
    e1d7:	41 bb 04 03 00 00    	mov    r11d,0x304
    e1dd:	48 89 84 24 c8 00 00 	mov    QWORD PTR [rsp+0xc8],rax
    e1e4:	00 
    e1e5:	41 6b c1 2a          	imul   eax,r9d,0x2a
    e1e9:	48 89 84 24 c0 00 00 	mov    QWORD PTR [rsp+0xc0],rax
    e1f0:	00 
    e1f1:	41 6b c1 2b          	imul   eax,r9d,0x2b
    e1f5:	48 89 84 24 b8 00 00 	mov    QWORD PTR [rsp+0xb8],rax
    e1fc:	00 
    e1fd:	8d 44 ad 00          	lea    eax,[rbp+rbp*4+0x0]
    e201:	48 89 84 24 b0 00 00 	mov    QWORD PTR [rsp+0xb0],rax
    e208:	00 
    e209:	41 6b c1 2c          	imul   eax,r9d,0x2c
    e20d:	48 89 84 24 a8 00 00 	mov    QWORD PTR [rsp+0xa8],rax
    e214:	00 
    e215:	41 6b c1 2e          	imul   eax,r9d,0x2e
    e219:	48 89 84 24 a0 00 00 	mov    QWORD PTR [rsp+0xa0],rax
    e220:	00 
    e221:	41 6b c1 2f          	imul   eax,r9d,0x2f
    e225:	48 89 84 24 98 00 00 	mov    QWORD PTR [rsp+0x98],rax
    e22c:	00 
    e22d:	43 8d 04 76          	lea    eax,[r14+r14*2]
    e231:	48 89 84 24 90 00 00 	mov    QWORD PTR [rsp+0x90],rax
    e238:	00 
    e239:	41 6b c1 31          	imul   eax,r9d,0x31
    e23d:	48 89 84 24 88 00 00 	mov    QWORD PTR [rsp+0x88],rax
    e244:	00 
    e245:	41 6b c1 32          	imul   eax,r9d,0x32
    e249:	48 89 84 24 80 00 00 	mov    QWORD PTR [rsp+0x80],rax
    e250:	00 
    e251:	41 6b c1 33          	imul   eax,r9d,0x33
    e255:	48 89 44 24 78       	mov    QWORD PTR [rsp+0x78],rax
    e25a:	41 6b c1 34          	imul   eax,r9d,0x34
    e25e:	48 89 44 24 70       	mov    QWORD PTR [rsp+0x70],rax
    e263:	41 6b c1 35          	imul   eax,r9d,0x35
    e267:	48 89 44 24 68       	mov    QWORD PTR [rsp+0x68],rax
    e26c:	41 6b c1 36          	imul   eax,r9d,0x36
    e270:	48 89 44 24 60       	mov    QWORD PTR [rsp+0x60],rax
    e275:	41 6b c1 37          	imul   eax,r9d,0x37
    e279:	48 89 44 24 58       	mov    QWORD PTR [rsp+0x58],rax
    e27e:	41 6b c1 38          	imul   eax,r9d,0x38
    e282:	48 89 44 24 50       	mov    QWORD PTR [rsp+0x50],rax
    e287:	41 6b c1 39          	imul   eax,r9d,0x39
    e28b:	48 89 44 24 48       	mov    QWORD PTR [rsp+0x48],rax
    e290:	41 6b c1 3a          	imul   eax,r9d,0x3a
    e294:	48 89 44 24 40       	mov    QWORD PTR [rsp+0x40],rax
    e299:	41 6b c1 3b          	imul   eax,r9d,0x3b
    e29d:	48 89 44 24 38       	mov    QWORD PTR [rsp+0x38],rax
    e2a2:	41 6b c1 3c          	imul   eax,r9d,0x3c
    e2a6:	48 89 44 24 30       	mov    QWORD PTR [rsp+0x30],rax
    e2ab:	41 6b c1 3d          	imul   eax,r9d,0x3d
    e2af:	48 89 44 24 28       	mov    QWORD PTR [rsp+0x28],rax
    e2b4:	44 89 c8             	mov    eax,r9d
    e2b7:	48 89 44 24 20       	mov    QWORD PTR [rsp+0x20],rax
    e2bc:	8b 44 24 84          	mov    eax,DWORD PTR [rsp-0x7c]
    e2c0:	48 89 44 24 c0       	mov    QWORD PTR [rsp-0x40],rax
    e2c5:	66 66 2e 0f 1f 84 00 	data16 cs nop WORD PTR [rax+rax*1+0x0]
    e2cc:	00 00 00 00 
    e2d0:	8b 44 24 84          	mov    eax,DWORD PTR [rsp-0x7c]
    e2d4:	0f af c7             	imul   eax,edi
    e2d7:	48 89 7c 24 b8       	mov    QWORD PTR [rsp-0x48],rdi
    e2dc:	c5 e0 57 db          	vxorps xmm3,xmm3,xmm3
    e2e0:	31 ff                	xor    edi,edi
    e2e2:	c5 e8 57 d2          	vxorps xmm2,xmm2,xmm2
    e2e6:	c5 f0 57 c9          	vxorps xmm1,xmm1,xmm1
    e2ea:	c5 f8 57 c0          	vxorps xmm0,xmm0,xmm0
    e2ee:	48 89 44 24 c8       	mov    QWORD PTR [rsp-0x38],rax
    e2f3:	31 c0                	xor    eax,eax
    e2f5:	66 66 2e 0f 1f 84 00 	data16 cs nop WORD PTR [rax+rax*1+0x0]
    e2fc:	00 00 00 00 
    e300:	89 44 24 d8          	mov    DWORD PTR [rsp-0x28],eax
    e304:	48 98                	cdqe   
    e306:	48 8d 8a 00 01 00 00 	lea    rcx,[rdx+0x100]
    e30d:	48 89 7c 24 d0       	mov    QWORD PTR [rsp-0x30],rdi
    e312:	45 31 e4             	xor    r12d,r12d
    e315:	4c 8d 34 81          	lea    r14,[rcx+rax*4]
    e319:	48 8b 44 24 c8       	mov    rax,QWORD PTR [rsp-0x38]
    e31e:	01 f8                	add    eax,edi
    e320:	6b e8 32             	imul   ebp,eax,0x32
    e323:	48 63 c5             	movsxd rax,ebp
    e326:	44 8d 7d 22          	lea    r15d,[rbp+0x22]
    e32a:	83 c5 02             	add    ebp,0x2
    e32d:	0f b6 0c 06          	movzx  ecx,BYTE PTR [rsi+rax*1]
    e331:	0f b6 44 06 01       	movzx  eax,BYTE PTR [rsi+rax*1+0x1]
    e336:	c1 e0 0a             	shl    eax,0xa
    e339:	48 03 04 24          	add    rax,QWORD PTR [rsp]
    e33d:	c5 fa 10 24 88       	vmovss xmm4,DWORD PTR [rax+rcx*4]
    e342:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
    e349:	1f 84 00 00 00 00 00 
    e350:	44 89 e7             	mov    edi,r12d
    e353:	c1 ef 05             	shr    edi,0x5
    e356:	c4 c2 28 f7 c4       	bextr  eax,r12d,r10d
    e35b:	45 31 d2             	xor    r10d,r10d
    e35e:	41 8d 0c 7f          	lea    ecx,[r15+rdi*2]
    e362:	8d 7c bd 00          	lea    edi,[rbp+rdi*4+0x0]
    e366:	48 63 c9             	movsxd rcx,ecx
    e369:	0f b6 1c 0e          	movzx  ebx,BYTE PTR [rsi+rcx*1]
    e36d:	0f b6 4c 0e 01       	movzx  ecx,BYTE PTR [rsi+rcx*1+0x1]
    e372:	84 c9                	test   cl,cl
    e374:	41 0f 99 c2          	setns  r10b
    e378:	01 c7                	add    edi,eax
    e37a:	c4 62 20 f7 c9       	bextr  r9d,ecx,r11d
    e37f:	c1 e1 08             	shl    ecx,0x8
    e382:	48 63 ff             	movsxd rdi,edi
    e385:	09 d9                	or     ecx,ebx
    e387:	8d 1c 40             	lea    ebx,[rax+rax*2]
    e38a:	47 8d 4c 09 01       	lea    r9d,[r9+r9*1+0x1]
    e38f:	c4 81 7a 10 74 95 00 	vmovss xmm6,DWORD PTR [r13+r10*4+0x0]
    e396:	41 ba 03 02 00 00    	mov    r10d,0x203
    e39c:	0f b6 04 3e          	movzx  eax,BYTE PTR [rsi+rdi*1]
    e3a0:	c4 e2 63 f7 f9       	shrx   edi,ecx,ebx
    e3a5:	c4 c1 02 2a e9       	vcvtsi2ss xmm5,xmm15,r9d
    e3aa:	45 89 e1             	mov    r9d,r12d
    e3ad:	41 83 e1 06          	and    r9d,0x6
    e3b1:	83 e7 07             	and    edi,0x7
    e3b4:	c1 e7 08             	shl    edi,0x8
    e3b7:	c5 da 59 ed          	vmulss xmm5,xmm4,xmm5
    e3bb:	09 c7                	or     edi,eax
    e3bd:	41 8d 3c f9          	lea    edi,[r9+rdi*8]
    e3c1:	41 0f b6 3c 38       	movzx  edi,BYTE PTR [r8+rdi*1]
    e3c6:	44 8d 8f 00 ff ff ff 	lea    r9d,[rdi-0x100]
    e3cd:	40 84 ff             	test   dil,dil
    e3d0:	44 0f 49 cf          	cmovns r9d,edi
    e3d4:	41 8d 7c 24 01       	lea    edi,[r12+0x1]
    e3d9:	c4 c1 02 2a f9       	vcvtsi2ss xmm7,xmm15,r9d
    e3de:	41 89 f9             	mov    r9d,edi
    e3e1:	83 e7 07             	and    edi,0x7
    e3e4:	41 c0 e9 03          	shr    r9b,0x3
    e3e8:	41 80 e1 03          	and    r9b,0x3
    e3ec:	c5 ca 58 ff          	vaddss xmm7,xmm6,xmm7
    e3f0:	45 0f b6 c9          	movzx  r9d,r9b
    e3f4:	47 8d 0c 49          	lea    r9d,[r9+r9*2]
    e3f8:	c5 d2 59 ff          	vmulss xmm7,xmm5,xmm7
    e3fc:	c4 e2 33 f7 c9       	shrx   ecx,ecx,r9d
    e401:	83 e1 07             	and    ecx,0x7
    e404:	c1 e1 08             	shl    ecx,0x8
    e407:	62 f2 7d 48 18 ff    	vbroadcastss zmm7,xmm7
    e40d:	62 51 44 48 59 46 ff 	vmulps zmm8,zmm7,ZMMWORD PTR [r14-0x40]
    e414:	09 c1                	or     ecx,eax
    e416:	62 51 44 48 59 4e fe 	vmulps zmm9,zmm7,ZMMWORD PTR [r14-0x80]
    e41d:	8d 04 cf             	lea    eax,[rdi+rcx*8]
    e420:	41 0f b6 04 00       	movzx  eax,BYTE PTR [r8+rax*1]
    e425:	8d 88 00 ff ff ff    	lea    ecx,[rax-0x100]
    e42b:	84 c0                	test   al,al
    e42d:	0f 49 c8             	cmovns ecx,eax
    e430:	49 83 c4 02          	add    r12,0x2
    e434:	62 d1 7c 48 58 c0    	vaddps zmm0,zmm0,zmm8
    e43a:	62 51 44 48 59 46 fd 	vmulps zmm8,zmm7,ZMMWORD PTR [r14-0xc0]
    e441:	62 d1 44 48 59 7e fc 	vmulps zmm7,zmm7,ZMMWORD PTR [r14-0x100]
    e448:	62 d1 74 48 58 c9    	vaddps zmm1,zmm1,zmm9
    e44e:	62 d1 6c 48 58 d0    	vaddps zmm2,zmm2,zmm8
    e454:	c5 02 2a c1          	vcvtsi2ss xmm8,xmm15,ecx
    e458:	62 f1 64 48 58 df    	vaddps zmm3,zmm3,zmm7
    e45e:	c5 ba 58 f6          	vaddss xmm6,xmm8,xmm6
    e462:	c5 d2 59 ee          	vmulss xmm5,xmm5,xmm6
    e466:	62 f2 7d 48 18 ed    	vbroadcastss zmm5,xmm5
    e46c:	62 d1 54 48 59 36    	vmulps zmm6,zmm5,ZMMWORD PTR [r14]
    e472:	62 51 54 48 59 46 01 	vmulps zmm8,zmm5,ZMMWORD PTR [r14+0x40]
    e479:	62 d1 54 48 59 7e 02 	vmulps zmm7,zmm5,ZMMWORD PTR [r14+0x80]
    e480:	62 d1 54 48 59 6e 03 	vmulps zmm5,zmm5,ZMMWORD PTR [r14+0xc0]
    e487:	49 81 c6 00 02 00 00 	add    r14,0x200
    e48e:	62 f1 64 48 58 de    	vaddps zmm3,zmm3,zmm6
    e494:	62 d1 6c 48 58 d0    	vaddps zmm2,zmm2,zmm8
    e49a:	62 f1 74 48 58 cf    	vaddps zmm1,zmm1,zmm7
    e4a0:	62 f1 7c 48 58 c5    	vaddps zmm0,zmm0,zmm5
    e4a6:	49 81 fc 00 01 00 00 	cmp    r12,0x100
    e4ad:	0f 85 9d fe ff ff    	jne    e350 <glm_vx_packed_batch_v2_iq1_s+0x410>
    e4b3:	48 8b 7c 24 d0       	mov    rdi,QWORD PTR [rsp-0x30]
    e4b8:	8b 44 24 d8          	mov    eax,DWORD PTR [rsp-0x28]
    e4bc:	48 ff c7             	inc    rdi
    e4bf:	05 00 40 00 00       	add    eax,0x4000
    e4c4:	48 3b 7c 24 c0       	cmp    rdi,QWORD PTR [rsp-0x40]
    e4c9:	0f 85 31 fe ff ff    	jne    e300 <glm_vx_packed_batch_v2_iq1_s+0x3c0>
    e4cf:	48 8b 7c 24 b8       	mov    rdi,QWORD PTR [rsp-0x48]
    e4d4:	4c 8b 4c 24 e8       	mov    r9,QWORD PTR [rsp-0x18]
    e4d9:	48 8b 4c 24 f8       	mov    rcx,QWORD PTR [rsp-0x8]
    e4de:	c4 e3 7d 19 dc 01    	vextractf128 xmm4,ymm3,0x1
    e4e4:	41 8d 04 39          	lea    eax,[r9+rdi*1]
    e4e8:	c5 fa 11 1c b9       	vmovss DWORD PTR [rcx+rdi*4],xmm3
    e4ed:	48 98                	cdqe   
    e4ef:	c4 e3 79 17 1c 81 01 	vextractps DWORD PTR [rcx+rax*4],xmm3,0x1
    e4f6:	48 8b 44 24 90       	mov    rax,QWORD PTR [rsp-0x70]
    e4fb:	01 f8                	add    eax,edi
    e4fd:	48 98                	cdqe   
    e4ff:	c4 e3 79 17 1c 81 02 	vextractps DWORD PTR [rcx+rax*4],xmm3,0x2
    e506:	48 8b 44 24 98       	mov    rax,QWORD PTR [rsp-0x68]
    e50b:	8d 04 38             	lea    eax,[rax+rdi*1]
    e50e:	48 98                	cdqe   
    e510:	c4 e3 79 17 1c 81 03 	vextractps DWORD PTR [rcx+rax*4],xmm3,0x3
    e517:	48 8b 44 24 b0       	mov    rax,QWORD PTR [rsp-0x50]
    e51c:	8d 04 38             	lea    eax,[rax+rdi*1]
    e51f:	48 98                	cdqe   
    e521:	c5 fa 11 24 81       	vmovss DWORD PTR [rcx+rax*4],xmm4
    e526:	48 8b 44 24 88       	mov    rax,QWORD PTR [rsp-0x78]
    e52b:	8d 04 38             	lea    eax,[rax+rdi*1]
    e52e:	48 98                	cdqe   
    e530:	c4 e3 79 17 24 81 01 	vextractps DWORD PTR [rcx+rax*4],xmm4,0x1
    e537:	48 8b 44 24 a8       	mov    rax,QWORD PTR [rsp-0x58]
    e53c:	8d 04 38             	lea    eax,[rax+rdi*1]
    e53f:	48 98                	cdqe   
    e541:	c4 e3 79 17 24 81 02 	vextractps DWORD PTR [rcx+rax*4],xmm4,0x2
    e548:	48 8b 44 24 a0       	mov    rax,QWORD PTR [rsp-0x60]
    e54d:	8d 04 38             	lea    eax,[rax+rdi*1]
    e550:	48 98                	cdqe   
    e552:	c4 e3 79 17 24 81 03 	vextractps DWORD PTR [rcx+rax*4],xmm4,0x3
    e559:	48 8b 44 24 e0       	mov    rax,QWORD PTR [rsp-0x20]
    e55e:	62 f3 7d 48 39 dc 02 	vextracti32x4 xmm4,zmm3,0x2
    e565:	62 f3 7d 48 39 db 03 	vextracti32x4 xmm3,zmm3,0x3
    e56c:	8d 04 38             	lea    eax,[rax+rdi*1]
    e56f:	48 98                	cdqe   
    e571:	c5 f9 7e 24 81       	vmovd  DWORD PTR [rcx+rax*4],xmm4
    e576:	48 8b 84 24 98 01 00 	mov    rax,QWORD PTR [rsp+0x198]
    e57d:	00 
    e57e:	8d 04 38             	lea    eax,[rax+rdi*1]
    e581:	48 98                	cdqe   
    e583:	c4 e3 79 17 24 81 01 	vextractps DWORD PTR [rcx+rax*4],xmm4,0x1
    e58a:	48 8b 84 24 90 01 00 	mov    rax,QWORD PTR [rsp+0x190]
    e591:	00 
    e592:	8d 04 38             	lea    eax,[rax+rdi*1]
    e595:	48 98                	cdqe   
    e597:	c4 e3 79 17 24 81 02 	vextractps DWORD PTR [rcx+rax*4],xmm4,0x2
    e59e:	48 8b 84 24 88 01 00 	mov    rax,QWORD PTR [rsp+0x188]
    e5a5:	00 
    e5a6:	8d 04 38             	lea    eax,[rax+rdi*1]
    e5a9:	48 98                	cdqe   
    e5ab:	c4 e3 79 17 24 81 03 	vextractps DWORD PTR [rcx+rax*4],xmm4,0x3
    e5b2:	48 8b 84 24 80 01 00 	mov    rax,QWORD PTR [rsp+0x180]
    e5b9:	00 
    e5ba:	8d 04 38             	lea    eax,[rax+rdi*1]
    e5bd:	48 98                	cdqe   
    e5bf:	c5 f9 7e 1c 81       	vmovd  DWORD PTR [rcx+rax*4],xmm3
    e5c4:	48 8b 84 24 78 01 00 	mov    rax,QWORD PTR [rsp+0x178]
    e5cb:	00 
    e5cc:	8d 04 38             	lea    eax,[rax+rdi*1]
    e5cf:	48 98                	cdqe   
    e5d1:	c4 e3 79 17 1c 81 01 	vextractps DWORD PTR [rcx+rax*4],xmm3,0x1
    e5d8:	48 8b 44 24 10       	mov    rax,QWORD PTR [rsp+0x10]
    e5dd:	8d 04 38             	lea    eax,[rax+rdi*1]
    e5e0:	48 98                	cdqe   
    e5e2:	c4 e3 79 17 1c 81 02 	vextractps DWORD PTR [rcx+rax*4],xmm3,0x2
    e5e9:	48 8b 84 24 70 01 00 	mov    rax,QWORD PTR [rsp+0x170]
    e5f0:	00 
    e5f1:	8d 04 38             	lea    eax,[rax+rdi*1]
    e5f4:	48 98                	cdqe   
    e5f6:	c4 e3 79 17 1c 81 03 	vextractps DWORD PTR [rcx+rax*4],xmm3,0x3
    e5fd:	48 8b 44 24 f0       	mov    rax,QWORD PTR [rsp-0x10]
    e602:	c4 e3 7d 19 d3 01    	vextractf128 xmm3,ymm2,0x1
    e608:	8d 04 38             	lea    eax,[rax+rdi*1]
    e60b:	48 98                	cdqe   
    e60d:	c5 fa 11 14 81       	vmovss DWORD PTR [rcx+rax*4],xmm2
    e612:	48 8b 84 24 68 01 00 	mov    rax,QWORD PTR [rsp+0x168]
    e619:	00 
    e61a:	8d 04 38             	lea    eax,[rax+rdi*1]
    e61d:	48 98                	cdqe   
    e61f:	c4 e3 79 17 14 81 01 	vextractps DWORD PTR [rcx+rax*4],xmm2,0x1
    e626:	48 8b 84 24 60 01 00 	mov    rax,QWORD PTR [rsp+0x160]
    e62d:	00 
    e62e:	8d 04 38             	lea    eax,[rax+rdi*1]
    e631:	48 98                	cdqe   
    e633:	c4 e3 79 17 14 81 02 	vextractps DWORD PTR [rcx+rax*4],xmm2,0x2
    e63a:	48 8b 84 24 58 01 00 	mov    rax,QWORD PTR [rsp+0x158]
    e641:	00 
    e642:	8d 04 38             	lea    eax,[rax+rdi*1]
    e645:	48 98                	cdqe   
    e647:	c4 e3 79 17 14 81 03 	vextractps DWORD PTR [rcx+rax*4],xmm2,0x3
    e64e:	48 8b 84 24 50 01 00 	mov    rax,QWORD PTR [rsp+0x150]
    e655:	00 
    e656:	8d 04 38             	lea    eax,[rax+rdi*1]
    e659:	48 98                	cdqe   
    e65b:	c5 fa 11 1c 81       	vmovss DWORD PTR [rcx+rax*4],xmm3
    e660:	48 8b 84 24 48 01 00 	mov    rax,QWORD PTR [rsp+0x148]
    e667:	00 
    e668:	8d 04 38             	lea    eax,[rax+rdi*1]
    e66b:	48 98                	cdqe   
    e66d:	c4 e3 79 17 1c 81 01 	vextractps DWORD PTR [rcx+rax*4],xmm3,0x1
    e674:	48 8b 84 24 30 01 00 	mov    rax,QWORD PTR [rsp+0x130]
    e67b:	00 
    e67c:	8d 04 38             	lea    eax,[rax+rdi*1]
    e67f:	48 98                	cdqe   
    e681:	c4 e3 79 17 1c 81 02 	vextractps DWORD PTR [rcx+rax*4],xmm3,0x2
    e688:	48 8b 44 24 18       	mov    rax,QWORD PTR [rsp+0x18]
    e68d:	8d 04 38             	lea    eax,[rax+rdi*1]
    e690:	48 98                	cdqe   
    e692:	c4 e3 79 17 1c 81 03 	vextractps DWORD PTR [rcx+rax*4],xmm3,0x3
    e699:	48 8b 84 24 40 01 00 	mov    rax,QWORD PTR [rsp+0x140]
    e6a0:	00 
    e6a1:	62 f3 7d 48 39 d3 02 	vextracti32x4 xmm3,zmm2,0x2
    e6a8:	62 f3 7d 48 39 d2 03 	vextracti32x4 xmm2,zmm2,0x3
    e6af:	8d 04 38             	lea    eax,[rax+rdi*1]
    e6b2:	48 98                	cdqe   
    e6b4:	c5 f9 7e 1c 81       	vmovd  DWORD PTR [rcx+rax*4],xmm3
    e6b9:	48 8b 84 24 38 01 00 	mov    rax,QWORD PTR [rsp+0x138]
    e6c0:	00 
    e6c1:	8d 04 38             	lea    eax,[rax+rdi*1]
    e6c4:	48 98                	cdqe   
    e6c6:	c4 e3 79 17 1c 81 01 	vextractps DWORD PTR [rcx+rax*4],xmm3,0x1
    e6cd:	48 8b 84 24 28 01 00 	mov    rax,QWORD PTR [rsp+0x128]
    e6d4:	00 
    e6d5:	8d 04 38             	lea    eax,[rax+rdi*1]
    e6d8:	48 98                	cdqe   
    e6da:	c4 e3 79 17 1c 81 02 	vextractps DWORD PTR [rcx+rax*4],xmm3,0x2
    e6e1:	48 8b 84 24 20 01 00 	mov    rax,QWORD PTR [rsp+0x120]
    e6e8:	00 
    e6e9:	8d 04 38             	lea    eax,[rax+rdi*1]
    e6ec:	48 98                	cdqe   
    e6ee:	c4 e3 79 17 1c 81 03 	vextractps DWORD PTR [rcx+rax*4],xmm3,0x3
    e6f5:	48 8b 84 24 18 01 00 	mov    rax,QWORD PTR [rsp+0x118]
    e6fc:	00 
    e6fd:	8d 04 38             	lea    eax,[rax+rdi*1]
    e700:	48 98                	cdqe   
    e702:	c5 f9 7e 14 81       	vmovd  DWORD PTR [rcx+rax*4],xmm2
    e707:	48 8b 84 24 10 01 00 	mov    rax,QWORD PTR [rsp+0x110]
    e70e:	00 
    e70f:	8d 04 38             	lea    eax,[rax+rdi*1]
    e712:	48 98                	cdqe   
    e714:	c4 e3 79 17 14 81 01 	vextractps DWORD PTR [rcx+rax*4],xmm2,0x1
    e71b:	48 8b 84 24 b8 01 00 	mov    rax,QWORD PTR [rsp+0x1b8]
    e722:	00 
    e723:	8d 04 38             	lea    eax,[rax+rdi*1]
    e726:	48 98                	cdqe   
    e728:	c4 e3 79 17 14 81 02 	vextractps DWORD PTR [rcx+rax*4],xmm2,0x2
    e72f:	48 8b 84 24 b0 01 00 	mov    rax,QWORD PTR [rsp+0x1b0]
    e736:	00 
    e737:	8d 04 38             	lea    eax,[rax+rdi*1]
    e73a:	48 98                	cdqe   
    e73c:	c4 e3 79 17 14 81 03 	vextractps DWORD PTR [rcx+rax*4],xmm2,0x3
    e743:	48 8b 84 24 c0 01 00 	mov    rax,QWORD PTR [rsp+0x1c0]
    e74a:	00 
    e74b:	c4 e3 7d 19 ca 01    	vextractf128 xmm2,ymm1,0x1
    e751:	8d 04 38             	lea    eax,[rax+rdi*1]
    e754:	48 98                	cdqe   
    e756:	c5 fa 11 0c 81       	vmovss DWORD PTR [rcx+rax*4],xmm1
    e75b:	48 8b 84 24 08 01 00 	mov    rax,QWORD PTR [rsp+0x108]
    e762:	00 
    e763:	8d 04 38             	lea    eax,[rax+rdi*1]
    e766:	48 98                	cdqe   
    e768:	c4 e3 79 17 0c 81 01 	vextractps DWORD PTR [rcx+rax*4],xmm1,0x1
    e76f:	48 8b 84 24 00 01 00 	mov    rax,QWORD PTR [rsp+0x100]
    e776:	00 
    e777:	8d 04 38             	lea    eax,[rax+rdi*1]
    e77a:	48 98                	cdqe   
    e77c:	c4 e3 79 17 0c 81 02 	vextractps DWORD PTR [rcx+rax*4],xmm1,0x2
    e783:	48 8b 84 24 f8 00 00 	mov    rax,QWORD PTR [rsp+0xf8]
    e78a:	00 
    e78b:	8d 04 38             	lea    eax,[rax+rdi*1]
    e78e:	48 98                	cdqe   
    e790:	c4 e3 79 17 0c 81 03 	vextractps DWORD PTR [rcx+rax*4],xmm1,0x3
    e797:	48 8b 84 24 f0 00 00 	mov    rax,QWORD PTR [rsp+0xf0]
    e79e:	00 
    e79f:	8d 04 38             	lea    eax,[rax+rdi*1]
    e7a2:	48 98                	cdqe   
    e7a4:	c5 fa 11 14 81       	vmovss DWORD PTR [rcx+rax*4],xmm2
    e7a9:	48 8b 84 24 e8 00 00 	mov    rax,QWORD PTR [rsp+0xe8]
    e7b0:	00 
    e7b1:	8d 04 38             	lea    eax,[rax+rdi*1]
    e7b4:	48 98                	cdqe   
    e7b6:	c4 e3 79 17 14 81 01 	vextractps DWORD PTR [rcx+rax*4],xmm2,0x1
    e7bd:	48 8b 84 24 e0 00 00 	mov    rax,QWORD PTR [rsp+0xe0]
    e7c4:	00 
    e7c5:	8d 04 38             	lea    eax,[rax+rdi*1]
    e7c8:	48 98                	cdqe   
    e7ca:	c4 e3 79 17 14 81 02 	vextractps DWORD PTR [rcx+rax*4],xmm2,0x2
    e7d1:	48 8b 84 24 d8 00 00 	mov    rax,QWORD PTR [rsp+0xd8]
    e7d8:	00 
    e7d9:	8d 04 38             	lea    eax,[rax+rdi*1]
    e7dc:	48 98                	cdqe   
    e7de:	c4 e3 79 17 14 81 03 	vextractps DWORD PTR [rcx+rax*4],xmm2,0x3
    e7e5:	48 8b 84 24 d0 00 00 	mov    rax,QWORD PTR [rsp+0xd0]
    e7ec:	00 
    e7ed:	62 f3 7d 48 39 ca 02 	vextracti32x4 xmm2,zmm1,0x2
    e7f4:	62 f3 7d 48 39 c9 03 	vextracti32x4 xmm1,zmm1,0x3
    e7fb:	8d 04 38             	lea    eax,[rax+rdi*1]
    e7fe:	48 98                	cdqe   
    e800:	c5 f9 7e 14 81       	vmovd  DWORD PTR [rcx+rax*4],xmm2
    e805:	48 8b 84 24 c8 00 00 	mov    rax,QWORD PTR [rsp+0xc8]
    e80c:	00 
    e80d:	8d 04 38             	lea    eax,[rax+rdi*1]
    e810:	48 98                	cdqe   
    e812:	c4 e3 79 17 14 81 01 	vextractps DWORD PTR [rcx+rax*4],xmm2,0x1
    e819:	48 8b 84 24 c0 00 00 	mov    rax,QWORD PTR [rsp+0xc0]
    e820:	00 
    e821:	8d 04 38             	lea    eax,[rax+rdi*1]
    e824:	48 98                	cdqe   
    e826:	c4 e3 79 17 14 81 02 	vextractps DWORD PTR [rcx+rax*4],xmm2,0x2
    e82d:	48 8b 84 24 b8 00 00 	mov    rax,QWORD PTR [rsp+0xb8]
    e834:	00 
    e835:	8d 04 38             	lea    eax,[rax+rdi*1]
    e838:	48 98                	cdqe   
    e83a:	c4 e3 79 17 14 81 03 	vextractps DWORD PTR [rcx+rax*4],xmm2,0x3
    e841:	48 8b 84 24 a8 00 00 	mov    rax,QWORD PTR [rsp+0xa8]
    e848:	00 
    e849:	8d 04 38             	lea    eax,[rax+rdi*1]
    e84c:	48 98                	cdqe   
    e84e:	c5 f9 7e 0c 81       	vmovd  DWORD PTR [rcx+rax*4],xmm1
    e853:	48 8b 84 24 b0 00 00 	mov    rax,QWORD PTR [rsp+0xb0]
    e85a:	00 
    e85b:	8d 04 38             	lea    eax,[rax+rdi*1]
    e85e:	48 98                	cdqe   
    e860:	c4 e3 79 17 0c 81 01 	vextractps DWORD PTR [rcx+rax*4],xmm1,0x1
    e867:	48 8b 84 24 a0 00 00 	mov    rax,QWORD PTR [rsp+0xa0]
    e86e:	00 
    e86f:	8d 04 38             	lea    eax,[rax+rdi*1]
    e872:	48 98                	cdqe   
    e874:	c4 e3 79 17 0c 81 02 	vextractps DWORD PTR [rcx+rax*4],xmm1,0x2
    e87b:	48 8b 84 24 98 00 00 	mov    rax,QWORD PTR [rsp+0x98]
    e882:	00 
    e883:	8d 04 38             	lea    eax,[rax+rdi*1]
    e886:	48 98                	cdqe   
    e888:	c4 e3 79 17 0c 81 03 	vextractps DWORD PTR [rcx+rax*4],xmm1,0x3
    e88f:	48 8b 84 24 90 00 00 	mov    rax,QWORD PTR [rsp+0x90]
    e896:	00 
    e897:	c4 e3 7d 19 c1 01    	vextractf128 xmm1,ymm0,0x1
    e89d:	8d 04 38             	lea    eax,[rax+rdi*1]
    e8a0:	48 98                	cdqe   
    e8a2:	c5 fa 11 04 81       	vmovss DWORD PTR [rcx+rax*4],xmm0
    e8a7:	48 8b 84 24 88 00 00 	mov    rax,QWORD PTR [rsp+0x88]
    e8ae:	00 
    e8af:	8d 04 38             	lea    eax,[rax+rdi*1]
    e8b2:	48 98                	cdqe   
    e8b4:	c4 e3 79 17 04 81 01 	vextractps DWORD PTR [rcx+rax*4],xmm0,0x1
    e8bb:	48 8b 84 24 80 00 00 	mov    rax,QWORD PTR [rsp+0x80]
    e8c2:	00 
    e8c3:	8d 04 38             	lea    eax,[rax+rdi*1]
    e8c6:	48 98                	cdqe   
    e8c8:	c4 e3 79 17 04 81 02 	vextractps DWORD PTR [rcx+rax*4],xmm0,0x2
    e8cf:	48 8b 44 24 78       	mov    rax,QWORD PTR [rsp+0x78]
    e8d4:	8d 04 38             	lea    eax,[rax+rdi*1]
    e8d7:	48 98                	cdqe   
    e8d9:	c4 e3 79 17 04 81 03 	vextractps DWORD PTR [rcx+rax*4],xmm0,0x3
    e8e0:	48 8b 44 24 70       	mov    rax,QWORD PTR [rsp+0x70]
    e8e5:	8d 04 38             	lea    eax,[rax+rdi*1]
    e8e8:	48 98                	cdqe   
    e8ea:	c5 fa 11 0c 81       	vmovss DWORD PTR [rcx+rax*4],xmm1
    e8ef:	48 8b 44 24 68       	mov    rax,QWORD PTR [rsp+0x68]
    e8f4:	8d 04 38             	lea    eax,[rax+rdi*1]
    e8f7:	48 98                	cdqe   
    e8f9:	c4 e3 79 17 0c 81 01 	vextractps DWORD PTR [rcx+rax*4],xmm1,0x1
    e900:	48 8b 44 24 60       	mov    rax,QWORD PTR [rsp+0x60]
    e905:	8d 04 38             	lea    eax,[rax+rdi*1]
    e908:	48 98                	cdqe   
    e90a:	c4 e3 79 17 0c 81 02 	vextractps DWORD PTR [rcx+rax*4],xmm1,0x2
    e911:	48 8b 44 24 58       	mov    rax,QWORD PTR [rsp+0x58]
    e916:	8d 04 38             	lea    eax,[rax+rdi*1]
    e919:	48 98                	cdqe   
    e91b:	c4 e3 79 17 0c 81 03 	vextractps DWORD PTR [rcx+rax*4],xmm1,0x3
    e922:	48 8b 44 24 50       	mov    rax,QWORD PTR [rsp+0x50]
    e927:	62 f3 7d 48 39 c1 02 	vextracti32x4 xmm1,zmm0,0x2
    e92e:	62 f3 7d 48 39 c0 03 	vextracti32x4 xmm0,zmm0,0x3
    e935:	8d 04 38             	lea    eax,[rax+rdi*1]
    e938:	48 98                	cdqe   
    e93a:	c5 f9 7e 0c 81       	vmovd  DWORD PTR [rcx+rax*4],xmm1
    e93f:	48 8b 44 24 48       	mov    rax,QWORD PTR [rsp+0x48]
    e944:	8d 04 38             	lea    eax,[rax+rdi*1]
    e947:	48 98                	cdqe   
    e949:	c4 e3 79 17 0c 81 01 	vextractps DWORD PTR [rcx+rax*4],xmm1,0x1
    e950:	48 8b 44 24 40       	mov    rax,QWORD PTR [rsp+0x40]
    e955:	8d 04 38             	lea    eax,[rax+rdi*1]
    e958:	48 98                	cdqe   
    e95a:	c4 e3 79 17 0c 81 02 	vextractps DWORD PTR [rcx+rax*4],xmm1,0x2
    e961:	48 8b 44 24 38       	mov    rax,QWORD PTR [rsp+0x38]
    e966:	8d 04 38             	lea    eax,[rax+rdi*1]
    e969:	48 98                	cdqe   
    e96b:	c4 e3 79 17 0c 81 03 	vextractps DWORD PTR [rcx+rax*4],xmm1,0x3
    e972:	48 8b 44 24 30       	mov    rax,QWORD PTR [rsp+0x30]
    e977:	8d 04 38             	lea    eax,[rax+rdi*1]
    e97a:	48 98                	cdqe   
    e97c:	c5 f9 7e 04 81       	vmovd  DWORD PTR [rcx+rax*4],xmm0
    e981:	48 8b 44 24 28       	mov    rax,QWORD PTR [rsp+0x28]
    e986:	01 f8                	add    eax,edi
    e988:	48 98                	cdqe   
    e98a:	c4 e3 79 17 04 81 01 	vextractps DWORD PTR [rcx+rax*4],xmm0,0x1
    e991:	48 8b 84 24 a0 01 00 	mov    rax,QWORD PTR [rsp+0x1a0]
    e998:	00 
    e999:	01 f8                	add    eax,edi
    e99b:	48 98                	cdqe   
    e99d:	c4 e3 79 17 04 81 02 	vextractps DWORD PTR [rcx+rax*4],xmm0,0x2
    e9a4:	48 8b 84 24 a8 01 00 	mov    rax,QWORD PTR [rsp+0x1a8]
    e9ab:	00 
    e9ac:	01 f8                	add    eax,edi
    e9ae:	48 ff c7             	inc    rdi
    e9b1:	48 98                	cdqe   
    e9b3:	c4 e3 79 17 04 81 03 	vextractps DWORD PTR [rcx+rax*4],xmm0,0x3
    e9ba:	48 3b 7c 24 20       	cmp    rdi,QWORD PTR [rsp+0x20]
    e9bf:	0f 85 0b f9 ff ff    	jne    e2d0 <glm_vx_packed_batch_v2_iq1_s+0x390>
    e9c5:	4c 8b 54 24 08       	mov    r10,QWORD PTR [rsp+0x8]
    e9ca:	44 8b 9c 24 08 02 00 	mov    r11d,DWORD PTR [rsp+0x208]
    e9d1:	00 
    e9d2:	8b 9c 24 00 02 00 00 	mov    ebx,DWORD PTR [rsp+0x200]
    e9d9:	41 be 40 00 00 00    	mov    r14d,0x40
    e9df:	45 29 f3             	sub    r11d,r14d
    e9e2:	44 89 5c 24 88       	mov    DWORD PTR [rsp-0x78],r11d
    e9e7:	41 83 fb 20          	cmp    r11d,0x20
    e9eb:	0f 8c 31 05 00 00    	jl     ef22 <glm_vx_packed_batch_v2_iq1_s+0xfe2>
    e9f1:	45 85 c9             	test   r9d,r9d
    e9f4:	0f 8e 1c 05 00 00    	jle    ef16 <glm_vx_packed_batch_v2_iq1_s+0xfd6>
    e9fa:	8b 44 24 84          	mov    eax,DWORD PTR [rsp-0x7c]
    e9fe:	c1 6c 24 88 05       	shr    DWORD PTR [rsp-0x78],0x5
    ea03:	62 d2 7d 28 7c c1    	vpbroadcastd ymm0,r9d
    ea09:	49 bd 40 e4 ff ff ff 	movabs r13,0xffffffffffffe440
    ea10:	ff ff ff 
    ea13:	89 df                	mov    edi,ebx
    ea15:	41 bf 03 02 00 00    	mov    r15d,0x203
    ea1b:	41 bb 04 03 00 00    	mov    r11d,0x304
    ea21:	83 f8 01             	cmp    eax,0x1
    ea24:	83 d0 00             	adc    eax,0x0
    ea27:	c1 e7 05             	shl    edi,0x5
    ea2a:	4d 01 d5             	add    r13,r10
    ea2d:	31 c9                	xor    ecx,ecx
    ea2f:	48 89 44 24 c0       	mov    QWORD PTR [rsp-0x40],rax
    ea34:	44 89 c8             	mov    eax,r9d
    ea37:	48 89 44 24 98       	mov    QWORD PTR [rsp-0x68],rax
    ea3c:	48 b8 a0 e3 ff ff ff 	movabs rax,0xffffffffffffe3a0
    ea43:	ff ff ff 
    ea46:	45 89 f1             	mov    r9d,r14d
    ea49:	44 0f af cb          	imul   r9d,ebx
    ea4d:	89 7c 24 a8          	mov    DWORD PTR [rsp-0x58],edi
    ea51:	c4 c1 7d 6f 0c 02    	vmovdqa ymm1,YMMWORD PTR [r10+rax*1]
    ea57:	48 b8 e0 e3 ff ff ff 	movabs rax,0xffffffffffffe3e0
    ea5e:	ff ff ff 
    ea61:	c4 c1 7d 6f 14 02    	vmovdqa ymm2,YMMWORD PTR [r10+rax*1]
    ea67:	48 b8 00 e4 ff ff ff 	movabs rax,0xffffffffffffe400
    ea6e:	ff ff ff 
    ea71:	c4 c1 7d 6f 1c 02    	vmovdqa ymm3,YMMWORD PTR [r10+rax*1]
    ea77:	48 b8 20 e4 ff ff ff 	movabs rax,0xffffffffffffe420
    ea7e:	ff ff ff 
    ea81:	44 89 4c 24 90       	mov    DWORD PTR [rsp-0x70],r9d
    ea86:	c4 c1 7d 6f 24 02    	vmovdqa ymm4,YMMWORD PTR [r10+rax*1]
    ea8c:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]
    ea90:	62 52 7d 28 7c c6    	vpbroadcastd ymm8,r14d
    ea96:	48 89 4c 24 b0       	mov    QWORD PTR [rsp-0x50],rcx
    ea9b:	4c 89 74 24 a0       	mov    QWORD PTR [rsp-0x60],r14
    eaa0:	45 31 c9             	xor    r9d,r9d
    eaa3:	c5 bd eb e9          	vpor   ymm5,ymm8,ymm1
    eaa7:	c5 bd eb f2          	vpor   ymm6,ymm8,ymm2
    eaab:	c5 bd eb fb          	vpor   ymm7,ymm8,ymm3
    eaaf:	c5 3d eb c4          	vpor   ymm8,ymm8,ymm4
    eab3:	c4 e2 55 40 e8       	vpmulld ymm5,ymm5,ymm0
    eab8:	c4 e2 4d 40 f0       	vpmulld ymm6,ymm6,ymm0
    eabd:	c4 e2 45 40 f8       	vpmulld ymm7,ymm7,ymm0
    eac2:	c4 62 3d 40 c0       	vpmulld ymm8,ymm8,ymm0
    eac7:	66 0f 1f 84 00 00 00 	nop    WORD PTR [rax+rax*1+0x0]
    eace:	00 00 
    ead0:	8b 7c 24 90          	mov    edi,DWORD PTR [rsp-0x70]
    ead4:	8b 44 24 84          	mov    eax,DWORD PTR [rsp-0x7c]
    ead8:	41 0f af c1          	imul   eax,r9d
    eadc:	4c 89 4c 24 b8       	mov    QWORD PTR [rsp-0x48],r9
    eae1:	c4 41 28 57 d2       	vxorps xmm10,xmm10,xmm10
    eae6:	31 c9                	xor    ecx,ecx
    eae8:	c4 41 30 57 c9       	vxorps xmm9,xmm9,xmm9
    eaed:	48 89 44 24 c8       	mov    QWORD PTR [rsp-0x38],rax
    eaf2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
    eaf9:	1f 84 00 00 00 00 00 
    eb00:	48 8b 44 24 c8       	mov    rax,QWORD PTR [rsp-0x38]
    eb05:	48 89 4c 24 d0       	mov    QWORD PTR [rsp-0x30],rcx
    eb0a:	89 7c 24 d8          	mov    DWORD PTR [rsp-0x28],edi
    eb0e:	31 db                	xor    ebx,ebx
    eb10:	01 c8                	add    eax,ecx
    eb12:	6b e8 32             	imul   ebp,eax,0x32
    eb15:	48 63 c5             	movsxd rax,ebp
    eb18:	44 8d 4d 22          	lea    r9d,[rbp+0x22]
    eb1c:	83 c5 02             	add    ebp,0x2
    eb1f:	0f b6 0c 06          	movzx  ecx,BYTE PTR [rsi+rax*1]
    eb23:	0f b6 44 06 01       	movzx  eax,BYTE PTR [rsi+rax*1+0x1]
    eb28:	c1 e0 0a             	shl    eax,0xa
    eb2b:	48 03 04 24          	add    rax,QWORD PTR [rsp]
    eb2f:	c5 7a 10 1c 88       	vmovss xmm11,DWORD PTR [rax+rcx*4]
    eb34:	66 66 66 2e 0f 1f 84 	data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
    eb3b:	00 00 00 00 00 
    eb40:	89 d8                	mov    eax,ebx
    eb42:	c1 e8 05             	shr    eax,0x5
    eb45:	c4 62 00 f7 f3       	bextr  r14d,ebx,r15d
    eb4a:	45 31 d2             	xor    r10d,r10d
    eb4d:	48 63 ff             	movsxd rdi,edi
    eb50:	41 8d 0c 41          	lea    ecx,[r9+rax*2]
    eb54:	8d 44 85 00          	lea    eax,[rbp+rax*4+0x0]
    eb58:	48 63 c9             	movsxd rcx,ecx
    eb5b:	44 0f b6 3c 0e       	movzx  r15d,BYTE PTR [rsi+rcx*1]
    eb60:	0f b6 4c 0e 01       	movzx  ecx,BYTE PTR [rsi+rcx*1+0x1]
    eb65:	84 c9                	test   cl,cl
    eb67:	41 89 cc             	mov    r12d,ecx
    eb6a:	41 0f 99 c2          	setns  r10b
    eb6e:	44 01 f0             	add    eax,r14d
    eb71:	41 c1 e4 08          	shl    r12d,0x8
    eb75:	47 8d 34 76          	lea    r14d,[r14+r14*2]
    eb79:	c4 e2 20 f7 c9       	bextr  ecx,ecx,r11d
    eb7e:	48 98                	cdqe   
    eb80:	45 09 fc             	or     r12d,r15d
    eb83:	8d 4c 09 01          	lea    ecx,[rcx+rcx*1+0x1]
    eb87:	c4 01 7a 10 74 95 00 	vmovss xmm14,DWORD PTR [r13+r10*4+0x0]
    eb8e:	41 bf 03 02 00 00    	mov    r15d,0x203
    eb94:	0f b6 04 06          	movzx  eax,BYTE PTR [rsi+rax*1]
    eb98:	c4 42 0b f7 f4       	shrx   r14d,r12d,r14d
    eb9d:	41 83 e6 07          	and    r14d,0x7
    eba1:	62 71 06 00 2a e1    	vcvtsi2ss xmm12,xmm31,ecx
    eba7:	89 d9                	mov    ecx,ebx
    eba9:	83 e1 06             	and    ecx,0x6
    ebac:	41 c1 e6 08          	shl    r14d,0x8
    ebb0:	c4 41 22 59 e4       	vmulss xmm12,xmm11,xmm12
    ebb5:	41 09 c6             	or     r14d,eax
    ebb8:	42 8d 0c f1          	lea    ecx,[rcx+r14*8]
    ebbc:	41 0f b6 0c 08       	movzx  ecx,BYTE PTR [r8+rcx*1]
    ebc1:	44 8d b1 00 ff ff ff 	lea    r14d,[rcx-0x100]
    ebc8:	84 c9                	test   cl,cl
    ebca:	44 0f 49 f1          	cmovns r14d,ecx
    ebce:	8d 4b 01             	lea    ecx,[rbx+0x1]
    ebd1:	41 89 ca             	mov    r10d,ecx
    ebd4:	83 e1 07             	and    ecx,0x7
    ebd7:	62 51 06 00 2a ee    	vcvtsi2ss xmm13,xmm31,r14d
    ebdd:	41 c0 ea 03          	shr    r10b,0x3
    ebe1:	41 80 e2 03          	and    r10b,0x3
    ebe5:	45 0f b6 d2          	movzx  r10d,r10b
    ebe9:	c4 41 0a 58 ed       	vaddss xmm13,xmm14,xmm13
    ebee:	47 8d 14 52          	lea    r10d,[r10+r10*2]
    ebf2:	c4 42 2b f7 d4       	shrx   r10d,r12d,r10d
    ebf7:	41 83 e2 07          	and    r10d,0x7
    ebfb:	41 c1 e2 08          	shl    r10d,0x8
    ebff:	41 09 c2             	or     r10d,eax
    ec02:	42 8d 04 d1          	lea    eax,[rcx+r10*8]
    ec06:	41 0f b6 04 00       	movzx  eax,BYTE PTR [r8+rax*1]
    ec0b:	8d 88 00 ff ff ff    	lea    ecx,[rax-0x100]
    ec11:	84 c0                	test   al,al
    ec13:	0f 49 c8             	cmovns ecx,eax
    ec16:	8d 47 20             	lea    eax,[rdi+0x20]
    ec19:	48 83 c3 02          	add    rbx,0x2
    ec1d:	62 71 06 00 2a f9    	vcvtsi2ss xmm15,xmm31,ecx
    ec23:	48 98                	cdqe   
    ec25:	c4 41 0a 58 ff       	vaddss xmm15,xmm14,xmm15
    ec2a:	c4 41 1a 59 ff       	vmulss xmm15,xmm12,xmm15
    ec2f:	c4 41 1a 59 e5       	vmulss xmm12,xmm12,xmm13
    ec34:	62 52 7d 48 18 e4    	vbroadcastss zmm12,xmm12
    ec3a:	62 71 1c 48 59 6c ba 	vmulps zmm13,zmm12,ZMMWORD PTR [rdx+rdi*4+0x40]
    ec41:	01 
    ec42:	62 71 1c 48 59 24 ba 	vmulps zmm12,zmm12,ZMMWORD PTR [rdx+rdi*4]
    ec49:	62 52 7d 48 18 ff    	vbroadcastss zmm15,xmm15
    ec4f:	62 e1 04 48 59 04 82 	vmulps zmm16,zmm15,ZMMWORD PTR [rdx+rax*4]
    ec56:	62 71 04 48 59 7c 82 	vmulps zmm15,zmm15,ZMMWORD PTR [rdx+rax*4+0x40]
    ec5d:	01 
    ec5e:	83 c7 40             	add    edi,0x40
    ec61:	62 51 34 48 58 cd    	vaddps zmm9,zmm9,zmm13
    ec67:	62 51 2c 48 58 d4    	vaddps zmm10,zmm10,zmm12
    ec6d:	62 31 2c 48 58 d0    	vaddps zmm10,zmm10,zmm16
    ec73:	62 51 34 48 58 cf    	vaddps zmm9,zmm9,zmm15
    ec79:	48 81 fb 00 01 00 00 	cmp    rbx,0x100
    ec80:	0f 85 ba fe ff ff    	jne    eb40 <glm_vx_packed_batch_v2_iq1_s+0xc00>
    ec86:	48 8b 4c 24 d0       	mov    rcx,QWORD PTR [rsp-0x30]
    ec8b:	8b 7c 24 d8          	mov    edi,DWORD PTR [rsp-0x28]
    ec8f:	48 ff c1             	inc    rcx
    ec92:	81 c7 00 20 00 00    	add    edi,0x2000
    ec98:	48 3b 4c 24 c0       	cmp    rcx,QWORD PTR [rsp-0x40]
    ec9d:	0f 85 5d fe ff ff    	jne    eb00 <glm_vx_packed_batch_v2_iq1_s+0xbc0>
    eca3:	4c 8b 4c 24 b8       	mov    r9,QWORD PTR [rsp-0x48]
    eca8:	48 8b 7c 24 f8       	mov    rdi,QWORD PTR [rsp-0x8]
    ecad:	c4 43 7d 19 d5 01    	vextractf128 xmm13,ymm10,0x1
    ecb3:	62 52 7d 28 7c d9    	vpbroadcastd ymm11,r9d
    ecb9:	49 ff c1             	inc    r9
    ecbc:	c5 25 fe e5          	vpaddd ymm12,ymm11,ymm5
    ecc0:	c5 79 7e e0          	vmovd  eax,xmm12
    ecc4:	c4 63 79 16 e1 01    	vpextrd ecx,xmm12,0x1
    ecca:	c4 43 79 16 e2 02    	vpextrd r10d,xmm12,0x2
    ecd0:	48 98                	cdqe   
    ecd2:	c5 7a 11 14 87       	vmovss DWORD PTR [rdi+rax*4],xmm10
    ecd7:	48 63 c1             	movsxd rax,ecx
    ecda:	c4 63 79 16 e1 03    	vpextrd ecx,xmm12,0x3
    ece0:	c4 43 7d 39 e4 01    	vextracti128 xmm12,ymm12,0x1
    ece6:	c4 63 79 17 14 87 01 	vextractps DWORD PTR [rdi+rax*4],xmm10,0x1
    eced:	49 63 c2             	movsxd rax,r10d
    ecf0:	c4 41 79 7e e2       	vmovd  r10d,xmm12
    ecf5:	c4 63 79 17 14 87 02 	vextractps DWORD PTR [rdi+rax*4],xmm10,0x2
    ecfc:	48 63 c1             	movsxd rax,ecx
    ecff:	c4 63 79 16 e1 01    	vpextrd ecx,xmm12,0x1
    ed05:	c4 63 79 17 14 87 03 	vextractps DWORD PTR [rdi+rax*4],xmm10,0x3
    ed0c:	49 63 c2             	movsxd rax,r10d
    ed0f:	c4 43 79 16 e2 03    	vpextrd r10d,xmm12,0x3
    ed15:	c5 7a 11 2c 87       	vmovss DWORD PTR [rdi+rax*4],xmm13
    ed1a:	48 63 c1             	movsxd rax,ecx
    ed1d:	c4 63 79 16 e1 02    	vpextrd ecx,xmm12,0x2
    ed23:	c5 25 fe e6          	vpaddd ymm12,ymm11,ymm6
    ed27:	c4 63 79 17 2c 87 01 	vextractps DWORD PTR [rdi+rax*4],xmm13,0x1
    ed2e:	48 63 c1             	movsxd rax,ecx
    ed31:	c4 63 79 16 e1 01    	vpextrd ecx,xmm12,0x1
    ed37:	c4 63 79 17 2c 87 02 	vextractps DWORD PTR [rdi+rax*4],xmm13,0x2
    ed3e:	49 63 c2             	movsxd rax,r10d
    ed41:	c4 43 79 16 e2 02    	vpextrd r10d,xmm12,0x2
    ed47:	c4 63 79 17 2c 87 03 	vextractps DWORD PTR [rdi+rax*4],xmm13,0x3
    ed4e:	c5 79 7e e0          	vmovd  eax,xmm12
    ed52:	62 53 7d 48 39 d5 02 	vextracti32x4 xmm13,zmm10,0x2
    ed59:	62 53 7d 48 39 d2 03 	vextracti32x4 xmm10,zmm10,0x3
    ed60:	48 98                	cdqe   
    ed62:	c5 79 7e 2c 87       	vmovd  DWORD PTR [rdi+rax*4],xmm13
    ed67:	48 63 c1             	movsxd rax,ecx
    ed6a:	c4 63 79 16 e1 03    	vpextrd ecx,xmm12,0x3
    ed70:	c4 43 7d 39 e4 01    	vextracti128 xmm12,ymm12,0x1
    ed76:	c4 63 79 17 2c 87 01 	vextractps DWORD PTR [rdi+rax*4],xmm13,0x1
    ed7d:	49 63 c2             	movsxd rax,r10d
    ed80:	c4 41 79 7e e2       	vmovd  r10d,xmm12
    ed85:	c4 63 79 17 2c 87 02 	vextractps DWORD PTR [rdi+rax*4],xmm13,0x2
    ed8c:	48 63 c1             	movsxd rax,ecx
    ed8f:	c4 63 79 16 e1 01    	vpextrd ecx,xmm12,0x1
    ed95:	c4 63 79 17 2c 87 03 	vextractps DWORD PTR [rdi+rax*4],xmm13,0x3
    ed9c:	49 63 c2             	movsxd rax,r10d
    ed9f:	c4 43 79 16 e2 03    	vpextrd r10d,xmm12,0x3
    eda5:	c5 79 7e 14 87       	vmovd  DWORD PTR [rdi+rax*4],xmm10
    edaa:	48 63 c1             	movsxd rax,ecx
    edad:	c4 63 79 16 e1 02    	vpextrd ecx,xmm12,0x2
    edb3:	c4 43 7d 19 cc 01    	vextractf128 xmm12,ymm9,0x1
    edb9:	c4 63 79 17 14 87 01 	vextractps DWORD PTR [rdi+rax*4],xmm10,0x1
    edc0:	48 63 c1             	movsxd rax,ecx
    edc3:	c4 63 79 17 14 87 02 	vextractps DWORD PTR [rdi+rax*4],xmm10,0x2
    edca:	49 63 c2             	movsxd rax,r10d
    edcd:	c4 63 79 17 14 87 03 	vextractps DWORD PTR [rdi+rax*4],xmm10,0x3
    edd4:	c5 25 fe d7          	vpaddd ymm10,ymm11,ymm7
    edd8:	c5 79 7e d0          	vmovd  eax,xmm10
    eddc:	c4 63 79 16 d1 01    	vpextrd ecx,xmm10,0x1
    ede2:	c4 43 79 16 d2 02    	vpextrd r10d,xmm10,0x2
    ede8:	48 98                	cdqe   
    edea:	c5 7a 11 0c 87       	vmovss DWORD PTR [rdi+rax*4],xmm9
    edef:	48 63 c1             	movsxd rax,ecx
    edf2:	c4 63 79 16 d1 03    	vpextrd ecx,xmm10,0x3
    edf8:	c4 43 7d 39 d2 01    	vextracti128 xmm10,ymm10,0x1
    edfe:	c4 63 79 17 0c 87 01 	vextractps DWORD PTR [rdi+rax*4],xmm9,0x1
    ee05:	49 63 c2             	movsxd rax,r10d
    ee08:	c4 41 79 7e d2       	vmovd  r10d,xmm10
    ee0d:	c4 63 79 17 0c 87 02 	vextractps DWORD PTR [rdi+rax*4],xmm9,0x2
    ee14:	48 63 c1             	movsxd rax,ecx
    ee17:	c4 63 79 16 d1 01    	vpextrd ecx,xmm10,0x1
    ee1d:	c4 63 79 17 0c 87 03 	vextractps DWORD PTR [rdi+rax*4],xmm9,0x3
    ee24:	49 63 c2             	movsxd rax,r10d
    ee27:	c4 43 79 16 d2 03    	vpextrd r10d,xmm10,0x3
    ee2d:	c5 7a 11 24 87       	vmovss DWORD PTR [rdi+rax*4],xmm12
    ee32:	48 63 c1             	movsxd rax,ecx
    ee35:	c4 63 79 16 d1 02    	vpextrd ecx,xmm10,0x2
    ee3b:	c4 41 3d fe d3       	vpaddd ymm10,ymm8,ymm11
    ee40:	62 53 7d 48 39 cb 02 	vextracti32x4 xmm11,zmm9,0x2
    ee47:	62 53 7d 48 39 c9 03 	vextracti32x4 xmm9,zmm9,0x3
    ee4e:	c4 63 79 17 24 87 01 	vextractps DWORD PTR [rdi+rax*4],xmm12,0x1
    ee55:	48 63 c1             	movsxd rax,ecx
    ee58:	c5 79 7e d1          	vmovd  ecx,xmm10
    ee5c:	c4 63 79 17 24 87 02 	vextractps DWORD PTR [rdi+rax*4],xmm12,0x2
    ee63:	49 63 c2             	movsxd rax,r10d
    ee66:	c4 43 79 16 d2 01    	vpextrd r10d,xmm10,0x1
    ee6c:	c4 63 79 17 24 87 03 	vextractps DWORD PTR [rdi+rax*4],xmm12,0x3
    ee73:	48 63 c1             	movsxd rax,ecx
    ee76:	c4 63 79 16 d1 02    	vpextrd ecx,xmm10,0x2
    ee7c:	c5 79 7e 1c 87       	vmovd  DWORD PTR [rdi+rax*4],xmm11
    ee81:	49 63 c2             	movsxd rax,r10d
    ee84:	c4 43 79 16 d2 03    	vpextrd r10d,xmm10,0x3
    ee8a:	c4 43 7d 39 d2 01    	vextracti128 xmm10,ymm10,0x1
    ee90:	c4 63 79 17 1c 87 01 	vextractps DWORD PTR [rdi+rax*4],xmm11,0x1
    ee97:	48 63 c1             	movsxd rax,ecx
    ee9a:	c5 79 7e d1          	vmovd  ecx,xmm10
    ee9e:	c4 63 79 17 1c 87 02 	vextractps DWORD PTR [rdi+rax*4],xmm11,0x2
    eea5:	49 63 c2             	movsxd rax,r10d
    eea8:	c4 43 79 16 d2 02    	vpextrd r10d,xmm10,0x2
    eeae:	c4 63 79 17 1c 87 03 	vextractps DWORD PTR [rdi+rax*4],xmm11,0x3
    eeb5:	48 63 c1             	movsxd rax,ecx
    eeb8:	c4 63 79 16 d1 01    	vpextrd ecx,xmm10,0x1
    eebe:	c5 79 7e 0c 87       	vmovd  DWORD PTR [rdi+rax*4],xmm9
    eec3:	48 63 c1             	movsxd rax,ecx
    eec6:	c4 63 79 16 d1 03    	vpextrd ecx,xmm10,0x3
    eecc:	c4 63 79 17 0c 87 01 	vextractps DWORD PTR [rdi+rax*4],xmm9,0x1
    eed3:	49 63 c2             	movsxd rax,r10d
    eed6:	c4 63 79 17 0c 87 02 	vextractps DWORD PTR [rdi+rax*4],xmm9,0x2
    eedd:	48 63 c1             	movsxd rax,ecx
    eee0:	c4 63 79 17 0c 87 03 	vextractps DWORD PTR [rdi+rax*4],xmm9,0x3
    eee7:	4c 3b 4c 24 98       	cmp    r9,QWORD PTR [rsp-0x68]
    eeec:	0f 85 de fb ff ff    	jne    ead0 <glm_vx_packed_batch_v2_iq1_s+0xb90>
    eef2:	8b 44 24 a8          	mov    eax,DWORD PTR [rsp-0x58]
    eef6:	4c 8b 74 24 a0       	mov    r14,QWORD PTR [rsp-0x60]
    eefb:	48 8b 4c 24 b0       	mov    rcx,QWORD PTR [rsp-0x50]
    ef00:	01 44 24 90          	add    DWORD PTR [rsp-0x70],eax
    ef04:	41 83 c6 20          	add    r14d,0x20
    ef08:	ff c1                	inc    ecx
    ef0a:	3b 4c 24 88          	cmp    ecx,DWORD PTR [rsp-0x78]
    ef0e:	0f 85 7c fb ff ff    	jne    ea90 <glm_vx_packed_batch_v2_iq1_s+0xb50>
    ef14:	eb 0c                	jmp    ef22 <glm_vx_packed_batch_v2_iq1_s+0xfe2>
    ef16:	8b 44 24 88          	mov    eax,DWORD PTR [rsp-0x78]
    ef1a:	25 e0 ff ff 7f       	and    eax,0x7fffffe0
    ef1f:	41 01 c6             	add    r14d,eax
    ef22:	8b 84 24 08 02 00 00 	mov    eax,DWORD PTR [rsp+0x208]
    ef29:	44 29 f0             	sub    eax,r14d
    ef2c:	89 44 24 88          	mov    DWORD PTR [rsp-0x78],eax
    ef30:	83 f8 10             	cmp    eax,0x10
    ef33:	0f 8c b7 03 00 00    	jl     f2f0 <glm_vx_packed_batch_v2_iq1_s+0x13b0>
    ef39:	48 8b 44 24 e8       	mov    rax,QWORD PTR [rsp-0x18]
    ef3e:	85 c0                	test   eax,eax
    ef40:	0f 8e 9e 03 00 00    	jle    f2e4 <glm_vx_packed_batch_v2_iq1_s+0x13a4>
    ef46:	8b 4c 24 84          	mov    ecx,DWORD PTR [rsp-0x7c]
    ef4a:	c1 6c 24 88 04       	shr    DWORD PTR [rsp-0x78],0x4
    ef4f:	62 f2 7d 28 7c c0    	vpbroadcastd ymm0,eax
    ef55:	4c 8b 54 24 08       	mov    r10,QWORD PTR [rsp+0x8]
    ef5a:	49 b9 40 e4 ff ff ff 	movabs r9,0xffffffffffffe440
    ef61:	ff ff ff 
    ef64:	41 bb 03 02 00 00    	mov    r11d,0x203
    ef6a:	41 bf 04 03 00 00    	mov    r15d,0x304
    ef70:	83 f9 01             	cmp    ecx,0x1
    ef73:	83 d1 00             	adc    ecx,0x0
    ef76:	4d 01 d1             	add    r9,r10
    ef79:	31 ff                	xor    edi,edi
    ef7b:	48 89 4c 24 c0       	mov    QWORD PTR [rsp-0x40],rcx
    ef80:	89 c1                	mov    ecx,eax
    ef82:	8b 84 24 00 02 00 00 	mov    eax,DWORD PTR [rsp+0x200]
    ef89:	48 89 4c 24 98       	mov    QWORD PTR [rsp-0x68],rcx
    ef8e:	44 89 f1             	mov    ecx,r14d
    ef91:	0f af c8             	imul   ecx,eax
    ef94:	c1 e0 04             	shl    eax,0x4
    ef97:	89 44 24 a8          	mov    DWORD PTR [rsp-0x58],eax
    ef9b:	48 b8 a0 e3 ff ff ff 	movabs rax,0xffffffffffffe3a0
    efa2:	ff ff ff 
    efa5:	c4 c1 7d 6f 0c 02    	vmovdqa ymm1,YMMWORD PTR [r10+rax*1]
    efab:	48 b8 e0 e3 ff ff ff 	movabs rax,0xffffffffffffe3e0
    efb2:	ff ff ff 
    efb5:	c4 c1 7d 6f 14 02    	vmovdqa ymm2,YMMWORD PTR [r10+rax*1]
    efbb:	89 4c 24 90          	mov    DWORD PTR [rsp-0x70],ecx
    efbf:	90                   	nop
    efc0:	62 d2 7d 28 7c e6    	vpbroadcastd ymm4,r14d
    efc6:	48 89 7c 24 b0       	mov    QWORD PTR [rsp-0x50],rdi
    efcb:	4c 89 74 24 a0       	mov    QWORD PTR [rsp-0x60],r14
    efd0:	45 31 d2             	xor    r10d,r10d
    efd3:	c5 dd fe d9          	vpaddd ymm3,ymm4,ymm1
    efd7:	c5 dd fe e2          	vpaddd ymm4,ymm4,ymm2
    efdb:	c4 e2 65 40 d8       	vpmulld ymm3,ymm3,ymm0
    efe0:	c4 e2 5d 40 e0       	vpmulld ymm4,ymm4,ymm0
    efe5:	66 66 2e 0f 1f 84 00 	data16 cs nop WORD PTR [rax+rax*1+0x0]
    efec:	00 00 00 00 
    eff0:	8b 44 24 84          	mov    eax,DWORD PTR [rsp-0x7c]
    eff4:	41 0f af c2          	imul   eax,r10d
    eff8:	4c 89 54 24 b8       	mov    QWORD PTR [rsp-0x48],r10
    effd:	44 8b 54 24 90       	mov    r10d,DWORD PTR [rsp-0x70]
    f002:	c5 d0 57 ed          	vxorps xmm5,xmm5,xmm5
    f006:	31 c9                	xor    ecx,ecx
    f008:	48 89 44 24 c8       	mov    QWORD PTR [rsp-0x38],rax
    f00d:	0f 1f 00             	nop    DWORD PTR [rax]
    f010:	48 8b 44 24 c8       	mov    rax,QWORD PTR [rsp-0x38]
    f015:	48 89 4c 24 d0       	mov    QWORD PTR [rsp-0x30],rcx
    f01a:	44 89 54 24 d8       	mov    DWORD PTR [rsp-0x28],r10d
    f01f:	45 31 e4             	xor    r12d,r12d
    f022:	01 c8                	add    eax,ecx
    f024:	6b f8 32             	imul   edi,eax,0x32
    f027:	48 63 c7             	movsxd rax,edi
    f02a:	8d 6f 22             	lea    ebp,[rdi+0x22]
    f02d:	83 c7 02             	add    edi,0x2
    f030:	0f b6 0c 06          	movzx  ecx,BYTE PTR [rsi+rax*1]
    f034:	0f b6 44 06 01       	movzx  eax,BYTE PTR [rsi+rax*1+0x1]
    f039:	c1 e0 0a             	shl    eax,0xa
    f03c:	48 03 04 24          	add    rax,QWORD PTR [rsp]
    f040:	c5 fa 10 34 88       	vmovss xmm6,DWORD PTR [rax+rcx*4]
    f045:	44 89 d1             	mov    ecx,r10d
    f048:	0f 1f 84 00 00 00 00 	nop    DWORD PTR [rax+rax*1+0x0]
    f04f:	00 
    f050:	45 89 e2             	mov    r10d,r12d
    f053:	41 c1 ea 05          	shr    r10d,0x5
    f057:	c4 42 20 f7 f4       	bextr  r14d,r12d,r11d
    f05c:	45 31 db             	xor    r11d,r11d
    f05f:	48 63 c9             	movsxd rcx,ecx
    f062:	42 8d 44 55 00       	lea    eax,[rbp+r10*2+0x0]
    f067:	46 8d 14 97          	lea    r10d,[rdi+r10*4]
    f06b:	48 98                	cdqe   
    f06d:	0f b6 1c 06          	movzx  ebx,BYTE PTR [rsi+rax*1]
    f071:	0f b6 44 06 01       	movzx  eax,BYTE PTR [rsi+rax*1+0x1]
    f076:	84 c0                	test   al,al
    f078:	41 89 c5             	mov    r13d,eax
    f07b:	41 0f 99 c3          	setns  r11b
    f07f:	45 01 f2             	add    r10d,r14d
    f082:	41 c1 e5 08          	shl    r13d,0x8
    f086:	c4 e2 00 f7 c0       	bextr  eax,eax,r15d
    f08b:	4d 63 d2             	movsxd r10,r10d
    f08e:	41 09 dd             	or     r13d,ebx
    f091:	43 8d 1c 76          	lea    ebx,[r14+r14*2]
    f095:	8d 44 00 01          	lea    eax,[rax+rax*1+0x1]
    f099:	c4 01 7a 10 0c 99    	vmovss xmm9,DWORD PTR [r9+r11*4]
    f09f:	41 bb 03 02 00 00    	mov    r11d,0x203
    f0a5:	46 0f b6 14 16       	movzx  r10d,BYTE PTR [rsi+r10*1]
    f0aa:	c4 c2 63 f7 dd       	shrx   ebx,r13d,ebx
    f0af:	c5 82 2a f8          	vcvtsi2ss xmm7,xmm15,eax
    f0b3:	44 89 e0             	mov    eax,r12d
    f0b6:	83 e0 06             	and    eax,0x6
    f0b9:	83 e3 07             	and    ebx,0x7
    f0bc:	c1 e3 08             	shl    ebx,0x8
    f0bf:	c5 ca 59 ff          	vmulss xmm7,xmm6,xmm7
    f0c3:	44 09 d3             	or     ebx,r10d
    f0c6:	8d 04 d8             	lea    eax,[rax+rbx*8]
    f0c9:	41 0f b6 04 00       	movzx  eax,BYTE PTR [r8+rax*1]
    f0ce:	8d 98 00 ff ff ff    	lea    ebx,[rax-0x100]
    f0d4:	84 c0                	test   al,al
    f0d6:	0f 49 d8             	cmovns ebx,eax
    f0d9:	41 8d 44 24 01       	lea    eax,[r12+0x1]
    f0de:	c5 02 2a c3          	vcvtsi2ss xmm8,xmm15,ebx
    f0e2:	89 c3                	mov    ebx,eax
    f0e4:	83 e0 07             	and    eax,0x7
    f0e7:	c0 eb 03             	shr    bl,0x3
    f0ea:	80 e3 03             	and    bl,0x3
    f0ed:	0f b6 db             	movzx  ebx,bl
    f0f0:	c4 41 32 58 c0       	vaddss xmm8,xmm9,xmm8
    f0f5:	8d 1c 5b             	lea    ebx,[rbx+rbx*2]
    f0f8:	c4 c2 63 f7 dd       	shrx   ebx,r13d,ebx
    f0fd:	83 e3 07             	and    ebx,0x7
    f100:	c1 e3 08             	shl    ebx,0x8
    f103:	44 09 d3             	or     ebx,r10d
    f106:	8d 04 d8             	lea    eax,[rax+rbx*8]
    f109:	41 0f b6 04 00       	movzx  eax,BYTE PTR [r8+rax*1]
    f10e:	44 8d 90 00 ff ff ff 	lea    r10d,[rax-0x100]
    f115:	84 c0                	test   al,al
    f117:	44 0f 49 d0          	cmovns r10d,eax
    f11b:	8d 41 10             	lea    eax,[rcx+0x10]
    f11e:	49 83 c4 02          	add    r12,0x2
    f122:	c4 41 02 2a d2       	vcvtsi2ss xmm10,xmm15,r10d
    f127:	48 98                	cdqe   
    f129:	c4 41 32 58 d2       	vaddss xmm10,xmm9,xmm10
    f12e:	c5 2a 59 d7          	vmulss xmm10,xmm10,xmm7
    f132:	c5 ba 59 ff          	vmulss xmm7,xmm8,xmm7
    f136:	62 f2 7d 48 18 ff    	vbroadcastss zmm7,xmm7
    f13c:	62 f1 44 48 59 3c 8a 	vmulps zmm7,zmm7,ZMMWORD PTR [rdx+rcx*4]
    f143:	62 52 7d 48 18 d2    	vbroadcastss zmm10,xmm10
    f149:	62 71 2c 48 59 14 82 	vmulps zmm10,zmm10,ZMMWORD PTR [rdx+rax*4]
    f150:	83 c1 20             	add    ecx,0x20
    f153:	62 f1 54 48 58 ef    	vaddps zmm5,zmm5,zmm7
    f159:	62 d1 54 48 58 ea    	vaddps zmm5,zmm5,zmm10
    f15f:	49 81 fc 00 01 00 00 	cmp    r12,0x100
    f166:	0f 85 e4 fe ff ff    	jne    f050 <glm_vx_packed_batch_v2_iq1_s+0x1110>
    f16c:	48 8b 4c 24 d0       	mov    rcx,QWORD PTR [rsp-0x30]
    f171:	44 8b 54 24 d8       	mov    r10d,DWORD PTR [rsp-0x28]
    f176:	48 ff c1             	inc    rcx
    f179:	41 81 c2 00 10 00 00 	add    r10d,0x1000
    f180:	48 3b 4c 24 c0       	cmp    rcx,QWORD PTR [rsp-0x40]
    f185:	0f 85 85 fe ff ff    	jne    f010 <glm_vx_packed_batch_v2_iq1_s+0x10d0>
    f18b:	4c 8b 54 24 b8       	mov    r10,QWORD PTR [rsp-0x48]
    f190:	48 8b 7c 24 f8       	mov    rdi,QWORD PTR [rsp-0x8]
    f195:	c4 c3 7d 19 e8 01    	vextractf128 xmm8,ymm5,0x1
    f19b:	62 d2 7d 28 7c f2    	vpbroadcastd ymm6,r10d
    f1a1:	49 ff c2             	inc    r10
    f1a4:	c5 e5 fe fe          	vpaddd ymm7,ymm3,ymm6
    f1a8:	c5 dd fe f6          	vpaddd ymm6,ymm4,ymm6
    f1ac:	c5 f9 7e f8          	vmovd  eax,xmm7
    f1b0:	c4 e3 79 16 f9 01    	vpextrd ecx,xmm7,0x1
    f1b6:	c4 e3 79 16 fb 02    	vpextrd ebx,xmm7,0x2
    f1bc:	48 98                	cdqe   
    f1be:	c5 fa 11 2c 87       	vmovss DWORD PTR [rdi+rax*4],xmm5
    f1c3:	48 63 c1             	movsxd rax,ecx
    f1c6:	c4 e3 79 16 f9 03    	vpextrd ecx,xmm7,0x3
    f1cc:	c4 e3 7d 39 ff 01    	vextracti128 xmm7,ymm7,0x1
    f1d2:	c4 e3 79 17 2c 87 01 	vextractps DWORD PTR [rdi+rax*4],xmm5,0x1
    f1d9:	48 63 c3             	movsxd rax,ebx
    f1dc:	c5 f9 7e fb          	vmovd  ebx,xmm7
    f1e0:	c4 e3 79 17 2c 87 02 	vextractps DWORD PTR [rdi+rax*4],xmm5,0x2
    f1e7:	48 63 c1             	movsxd rax,ecx
    f1ea:	c4 e3 79 16 f9 01    	vpextrd ecx,xmm7,0x1
    f1f0:	c4 e3 79 17 2c 87 03 	vextractps DWORD PTR [rdi+rax*4],xmm5,0x3
    f1f7:	48 63 c3             	movsxd rax,ebx
    f1fa:	c4 e3 79 16 fb 03    	vpextrd ebx,xmm7,0x3
    f200:	c5 7a 11 04 87       	vmovss DWORD PTR [rdi+rax*4],xmm8
    f205:	48 63 c1             	movsxd rax,ecx
    f208:	c4 e3 79 16 f9 02    	vpextrd ecx,xmm7,0x2
    f20e:	62 f3 7d 48 39 ef 02 	vextracti32x4 xmm7,zmm5,0x2
    f215:	62 f3 7d 48 39 ed 03 	vextracti32x4 xmm5,zmm5,0x3
    f21c:	c4 63 79 17 04 87 01 	vextractps DWORD PTR [rdi+rax*4],xmm8,0x1
    f223:	48 63 c1             	movsxd rax,ecx
    f226:	c5 f9 7e f1          	vmovd  ecx,xmm6
    f22a:	c4 63 79 17 04 87 02 	vextractps DWORD PTR [rdi+rax*4],xmm8,0x2
    f231:	48 63 c3             	movsxd rax,ebx
    f234:	c4 e3 79 16 f3 01    	vpextrd ebx,xmm6,0x1
    f23a:	c4 63 79 17 04 87 03 	vextractps DWORD PTR [rdi+rax*4],xmm8,0x3
    f241:	48 63 c1             	movsxd rax,ecx
    f244:	c4 e3 79 16 f1 02    	vpextrd ecx,xmm6,0x2
    f24a:	c5 f9 7e 3c 87       	vmovd  DWORD PTR [rdi+rax*4],xmm7
    f24f:	48 63 c3             	movsxd rax,ebx
    f252:	c4 e3 79 16 f3 03    	vpextrd ebx,xmm6,0x3
    f258:	c4 e3 7d 39 f6 01    	vextracti128 xmm6,ymm6,0x1
    f25e:	c4 e3 79 17 3c 87 01 	vextractps DWORD PTR [rdi+rax*4],xmm7,0x1
    f265:	48 63 c1             	movsxd rax,ecx
    f268:	c5 f9 7e f1          	vmovd  ecx,xmm6
    f26c:	c4 e3 79 17 3c 87 02 	vextractps DWORD PTR [rdi+rax*4],xmm7,0x2
    f273:	48 63 c3             	movsxd rax,ebx
    f276:	c4 e3 79 16 f3 02    	vpextrd ebx,xmm6,0x2
    f27c:	c4 e3 79 17 3c 87 03 	vextractps DWORD PTR [rdi+rax*4],xmm7,0x3
    f283:	48 63 c1             	movsxd rax,ecx
    f286:	c4 e3 79 16 f1 01    	vpextrd ecx,xmm6,0x1
    f28c:	c5 f9 7e 2c 87       	vmovd  DWORD PTR [rdi+rax*4],xmm5
    f291:	48 63 c1             	movsxd rax,ecx
    f294:	c4 e3 79 16 f1 03    	vpextrd ecx,xmm6,0x3
    f29a:	c4 e3 79 17 2c 87 01 	vextractps DWORD PTR [rdi+rax*4],xmm5,0x1
    f2a1:	48 63 c3             	movsxd rax,ebx
    f2a4:	c4 e3 79 17 2c 87 02 	vextractps DWORD PTR [rdi+rax*4],xmm5,0x2
    f2ab:	48 63 c1             	movsxd rax,ecx
    f2ae:	c4 e3 79 17 2c 87 03 	vextractps DWORD PTR [rdi+rax*4],xmm5,0x3
    f2b5:	4c 3b 54 24 98       	cmp    r10,QWORD PTR [rsp-0x68]
    f2ba:	0f 85 30 fd ff ff    	jne    eff0 <glm_vx_packed_batch_v2_iq1_s+0x10b0>
    f2c0:	8b 44 24 a8          	mov    eax,DWORD PTR [rsp-0x58]
    f2c4:	4c 8b 74 24 a0       	mov    r14,QWORD PTR [rsp-0x60]
    f2c9:	48 8b 7c 24 b0       	mov    rdi,QWORD PTR [rsp-0x50]
    f2ce:	01 44 24 90          	add    DWORD PTR [rsp-0x70],eax
    f2d2:	41 83 c6 10          	add    r14d,0x10
    f2d6:	ff c7                	inc    edi
    f2d8:	3b 7c 24 88          	cmp    edi,DWORD PTR [rsp-0x78]
    f2dc:	0f 85 de fc ff ff    	jne    efc0 <glm_vx_packed_batch_v2_iq1_s+0x1080>
    f2e2:	eb 0c                	jmp    f2f0 <glm_vx_packed_batch_v2_iq1_s+0x13b0>
    f2e4:	8b 44 24 88          	mov    eax,DWORD PTR [rsp-0x78]
    f2e8:	25 f0 ff ff 7f       	and    eax,0x7ffffff0
    f2ed:	41 01 c6             	add    r14d,eax
    f2f0:	8b 84 24 08 02 00 00 	mov    eax,DWORD PTR [rsp+0x208]
    f2f7:	44 29 f0             	sub    eax,r14d
    f2fa:	89 44 24 88          	mov    DWORD PTR [rsp-0x78],eax
    f2fe:	83 f8 08             	cmp    eax,0x8
    f301:	0f 8c 0a 03 00 00    	jl     f611 <glm_vx_packed_batch_v2_iq1_s+0x16d1>
    f307:	48 8b 44 24 e8       	mov    rax,QWORD PTR [rsp-0x18]
    f30c:	85 c0                	test   eax,eax
    f30e:	0f 8e f1 02 00 00    	jle    f605 <glm_vx_packed_batch_v2_iq1_s+0x16c5>
    f314:	8b 4c 24 84          	mov    ecx,DWORD PTR [rsp-0x7c]
    f318:	c1 6c 24 88 03       	shr    DWORD PTR [rsp-0x78],0x3
    f31d:	62 f2 7d 28 7c c0    	vpbroadcastd ymm0,eax
    f323:	4c 8b 54 24 08       	mov    r10,QWORD PTR [rsp+0x8]
    f328:	49 b9 40 e4 ff ff ff 	movabs r9,0xffffffffffffe440
    f32f:	ff ff ff 
    f332:	41 bb 03 02 00 00    	mov    r11d,0x203
    f338:	41 bf 04 03 00 00    	mov    r15d,0x304
    f33e:	83 f9 01             	cmp    ecx,0x1
    f341:	83 d1 00             	adc    ecx,0x0
    f344:	4d 01 d1             	add    r9,r10
    f347:	31 ff                	xor    edi,edi
    f349:	48 89 4c 24 c0       	mov    QWORD PTR [rsp-0x40],rcx
    f34e:	89 c1                	mov    ecx,eax
    f350:	8b 84 24 00 02 00 00 	mov    eax,DWORD PTR [rsp+0x200]
    f357:	48 89 4c 24 98       	mov    QWORD PTR [rsp-0x68],rcx
    f35c:	44 89 f1             	mov    ecx,r14d
    f35f:	0f af c8             	imul   ecx,eax
    f362:	8d 04 c5 00 00 00 00 	lea    eax,[rax*8+0x0]
    f369:	89 44 24 a8          	mov    DWORD PTR [rsp-0x58],eax
    f36d:	48 b8 a0 e3 ff ff ff 	movabs rax,0xffffffffffffe3a0
    f374:	ff ff ff 
    f377:	c4 c1 7d 6f 0c 02    	vmovdqa ymm1,YMMWORD PTR [r10+rax*1]
    f37d:	89 4c 24 90          	mov    DWORD PTR [rsp-0x70],ecx
    f381:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
    f388:	0f 1f 84 00 00 00 00 
    f38f:	00 
    f390:	62 d2 7d 28 7c d6    	vpbroadcastd ymm2,r14d
    f396:	48 89 7c 24 b0       	mov    QWORD PTR [rsp-0x50],rdi
    f39b:	4c 89 74 24 a0       	mov    QWORD PTR [rsp-0x60],r14
    f3a0:	45 31 d2             	xor    r10d,r10d
    f3a3:	c5 ed fe d1          	vpaddd ymm2,ymm2,ymm1
    f3a7:	c4 e2 6d 40 d0       	vpmulld ymm2,ymm2,ymm0
    f3ac:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]
    f3b0:	8b 44 24 84          	mov    eax,DWORD PTR [rsp-0x7c]
    f3b4:	41 0f af c2          	imul   eax,r10d
    f3b8:	4c 89 54 24 b8       	mov    QWORD PTR [rsp-0x48],r10
    f3bd:	44 8b 54 24 90       	mov    r10d,DWORD PTR [rsp-0x70]
    f3c2:	c5 e1 ef db          	vpxor  xmm3,xmm3,xmm3
    f3c6:	31 c9                	xor    ecx,ecx
    f3c8:	48 89 44 24 c8       	mov    QWORD PTR [rsp-0x38],rax
    f3cd:	0f 1f 00             	nop    DWORD PTR [rax]
    f3d0:	48 8b 44 24 c8       	mov    rax,QWORD PTR [rsp-0x38]
    f3d5:	48 89 4c 24 d0       	mov    QWORD PTR [rsp-0x30],rcx
    f3da:	44 89 54 24 d8       	mov    DWORD PTR [rsp-0x28],r10d
    f3df:	45 31 f6             	xor    r14d,r14d
    f3e2:	01 c8                	add    eax,ecx
    f3e4:	6b f8 32             	imul   edi,eax,0x32
    f3e7:	48 63 c7             	movsxd rax,edi
    f3ea:	44 8d 67 22          	lea    r12d,[rdi+0x22]
    f3ee:	83 c7 02             	add    edi,0x2
    f3f1:	0f b6 0c 06          	movzx  ecx,BYTE PTR [rsi+rax*1]
    f3f5:	0f b6 44 06 01       	movzx  eax,BYTE PTR [rsi+rax*1+0x1]
    f3fa:	c1 e0 0a             	shl    eax,0xa
    f3fd:	48 03 04 24          	add    rax,QWORD PTR [rsp]
    f401:	c5 fa 10 24 88       	vmovss xmm4,DWORD PTR [rax+rcx*4]
    f406:	44 89 d1             	mov    ecx,r10d
    f409:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]
    f410:	45 89 f2             	mov    r10d,r14d
    f413:	41 c1 ea 05          	shr    r10d,0x5
    f417:	c4 c2 20 f7 ee       	bextr  ebp,r14d,r11d
    f41c:	45 31 db             	xor    r11d,r11d
    f41f:	48 63 c9             	movsxd rcx,ecx
    f422:	43 8d 04 54          	lea    eax,[r12+r10*2]
    f426:	46 8d 14 97          	lea    r10d,[rdi+r10*4]
    f42a:	48 98                	cdqe   
    f42c:	0f b6 1c 06          	movzx  ebx,BYTE PTR [rsi+rax*1]
    f430:	0f b6 44 06 01       	movzx  eax,BYTE PTR [rsi+rax*1+0x1]
    f435:	84 c0                	test   al,al
    f437:	41 89 c5             	mov    r13d,eax
    f43a:	41 0f 99 c3          	setns  r11b
    f43e:	41 01 ea             	add    r10d,ebp
    f441:	41 c1 e5 08          	shl    r13d,0x8
    f445:	c4 e2 00 f7 c0       	bextr  eax,eax,r15d
    f44a:	4d 63 d2             	movsxd r10,r10d
    f44d:	41 09 dd             	or     r13d,ebx
    f450:	8d 5c 6d 00          	lea    ebx,[rbp+rbp*2+0x0]
    f454:	8d 44 00 01          	lea    eax,[rax+rax*1+0x1]
    f458:	c4 81 7a 10 3c 99    	vmovss xmm7,DWORD PTR [r9+r11*4]
    f45e:	41 bb 03 02 00 00    	mov    r11d,0x203
    f464:	46 0f b6 14 16       	movzx  r10d,BYTE PTR [rsi+r10*1]
    f469:	c4 c2 63 f7 dd       	shrx   ebx,r13d,ebx
    f46e:	c5 82 2a e8          	vcvtsi2ss xmm5,xmm15,eax
    f472:	44 89 f0             	mov    eax,r14d
    f475:	83 e0 06             	and    eax,0x6
    f478:	83 e3 07             	and    ebx,0x7
    f47b:	c1 e3 08             	shl    ebx,0x8
    f47e:	c5 da 59 ed          	vmulss xmm5,xmm4,xmm5
    f482:	44 09 d3             	or     ebx,r10d
    f485:	8d 04 d8             	lea    eax,[rax+rbx*8]
    f488:	41 0f b6 04 00       	movzx  eax,BYTE PTR [r8+rax*1]
    f48d:	8d 98 00 ff ff ff    	lea    ebx,[rax-0x100]
    f493:	84 c0                	test   al,al
    f495:	0f 49 d8             	cmovns ebx,eax
    f498:	41 8d 46 01          	lea    eax,[r14+0x1]
    f49c:	c5 82 2a f3          	vcvtsi2ss xmm6,xmm15,ebx
    f4a0:	89 c3                	mov    ebx,eax
    f4a2:	83 e0 07             	and    eax,0x7
    f4a5:	c0 eb 03             	shr    bl,0x3
    f4a8:	80 e3 03             	and    bl,0x3
    f4ab:	0f b6 db             	movzx  ebx,bl
    f4ae:	c5 c2 58 f6          	vaddss xmm6,xmm7,xmm6
    f4b2:	8d 1c 5b             	lea    ebx,[rbx+rbx*2]
    f4b5:	c4 c2 63 f7 dd       	shrx   ebx,r13d,ebx
    f4ba:	83 e3 07             	and    ebx,0x7
    f4bd:	c1 e3 08             	shl    ebx,0x8
    f4c0:	44 09 d3             	or     ebx,r10d
    f4c3:	8d 04 d8             	lea    eax,[rax+rbx*8]
    f4c6:	41 0f b6 04 00       	movzx  eax,BYTE PTR [r8+rax*1]
    f4cb:	44 8d 90 00 ff ff ff 	lea    r10d,[rax-0x100]
    f4d2:	84 c0                	test   al,al
    f4d4:	44 0f 49 d0          	cmovns r10d,eax
    f4d8:	8d 41 08             	lea    eax,[rcx+0x8]
    f4db:	49 83 c6 02          	add    r14,0x2
    f4df:	c4 41 02 2a c2       	vcvtsi2ss xmm8,xmm15,r10d
    f4e4:	48 98                	cdqe   
    f4e6:	c5 3a 58 c7          	vaddss xmm8,xmm8,xmm7
    f4ea:	c5 3a 59 c5          	vmulss xmm8,xmm8,xmm5
    f4ee:	c5 d2 59 ee          	vmulss xmm5,xmm5,xmm6
    f4f2:	c4 e2 7d 18 ed       	vbroadcastss ymm5,xmm5
    f4f7:	c5 d4 59 2c 8a       	vmulps ymm5,ymm5,YMMWORD PTR [rdx+rcx*4]
    f4fc:	c4 42 7d 18 c0       	vbroadcastss ymm8,xmm8
    f501:	c5 3c 59 04 82       	vmulps ymm8,ymm8,YMMWORD PTR [rdx+rax*4]
    f506:	83 c1 10             	add    ecx,0x10
    f509:	c5 e4 58 dd          	vaddps ymm3,ymm3,ymm5
    f50d:	c5 bc 58 db          	vaddps ymm3,ymm8,ymm3
    f511:	49 81 fe 00 01 00 00 	cmp    r14,0x100
    f518:	0f 85 f2 fe ff ff    	jne    f410 <glm_vx_packed_batch_v2_iq1_s+0x14d0>
    f51e:	48 8b 4c 24 d0       	mov    rcx,QWORD PTR [rsp-0x30]
    f523:	44 8b 54 24 d8       	mov    r10d,DWORD PTR [rsp-0x28]
    f528:	48 ff c1             	inc    rcx
    f52b:	41 81 c2 00 08 00 00 	add    r10d,0x800
    f532:	48 3b 4c 24 c0       	cmp    rcx,QWORD PTR [rsp-0x40]
    f537:	0f 85 93 fe ff ff    	jne    f3d0 <glm_vx_packed_batch_v2_iq1_s+0x1490>
    f53d:	4c 8b 54 24 b8       	mov    r10,QWORD PTR [rsp-0x48]
    f542:	48 8b 7c 24 f8       	mov    rdi,QWORD PTR [rsp-0x8]
    f547:	62 d2 7d 28 7c e2    	vpbroadcastd ymm4,r10d
    f54d:	49 ff c2             	inc    r10
    f550:	c5 ed fe e4          	vpaddd ymm4,ymm2,ymm4
    f554:	c5 f9 7e e0          	vmovd  eax,xmm4
    f558:	c4 e3 79 16 e1 01    	vpextrd ecx,xmm4,0x1
    f55e:	c4 e3 79 16 e3 02    	vpextrd ebx,xmm4,0x2
    f564:	48 98                	cdqe   
    f566:	c5 fa 11 1c 87       	vmovss DWORD PTR [rdi+rax*4],xmm3
    f56b:	48 63 c1             	movsxd rax,ecx
    f56e:	c4 e3 79 16 e1 03    	vpextrd ecx,xmm4,0x3
    f574:	c4 e3 7d 39 e4 01    	vextracti128 xmm4,ymm4,0x1
    f57a:	c4 e3 79 17 1c 87 01 	vextractps DWORD PTR [rdi+rax*4],xmm3,0x1
    f581:	48 63 c3             	movsxd rax,ebx
    f584:	c4 e3 79 16 e3 02    	vpextrd ebx,xmm4,0x2
    f58a:	c4 e3 79 17 1c 87 02 	vextractps DWORD PTR [rdi+rax*4],xmm3,0x2
    f591:	48 63 c1             	movsxd rax,ecx
    f594:	c4 e3 79 16 e1 01    	vpextrd ecx,xmm4,0x1
    f59a:	c4 e3 79 17 1c 87 03 	vextractps DWORD PTR [rdi+rax*4],xmm3,0x3
    f5a1:	c5 f9 7e e0          	vmovd  eax,xmm4
    f5a5:	c4 e3 7d 39 db 01    	vextracti128 xmm3,ymm3,0x1
    f5ab:	48 98                	cdqe   
    f5ad:	c5 f9 7e 1c 87       	vmovd  DWORD PTR [rdi+rax*4],xmm3
    f5b2:	48 63 c1             	movsxd rax,ecx
    f5b5:	c4 e3 79 16 e1 03    	vpextrd ecx,xmm4,0x3
    f5bb:	c4 e3 79 17 1c 87 01 	vextractps DWORD PTR [rdi+rax*4],xmm3,0x1
    f5c2:	48 63 c3             	movsxd rax,ebx
    f5c5:	c4 e3 79 17 1c 87 02 	vextractps DWORD PTR [rdi+rax*4],xmm3,0x2
    f5cc:	48 63 c1             	movsxd rax,ecx
    f5cf:	c4 e3 79 17 1c 87 03 	vextractps DWORD PTR [rdi+rax*4],xmm3,0x3
    f5d6:	4c 3b 54 24 98       	cmp    r10,QWORD PTR [rsp-0x68]
    f5db:	0f 85 cf fd ff ff    	jne    f3b0 <glm_vx_packed_batch_v2_iq1_s+0x1470>
    f5e1:	8b 44 24 a8          	mov    eax,DWORD PTR [rsp-0x58]
    f5e5:	4c 8b 74 24 a0       	mov    r14,QWORD PTR [rsp-0x60]
    f5ea:	48 8b 7c 24 b0       	mov    rdi,QWORD PTR [rsp-0x50]
    f5ef:	01 44 24 90          	add    DWORD PTR [rsp-0x70],eax
    f5f3:	41 83 c6 08          	add    r14d,0x8
    f5f7:	ff c7                	inc    edi
    f5f9:	3b 7c 24 88          	cmp    edi,DWORD PTR [rsp-0x78]
    f5fd:	0f 85 8d fd ff ff    	jne    f390 <glm_vx_packed_batch_v2_iq1_s+0x1450>
    f603:	eb 0c                	jmp    f611 <glm_vx_packed_batch_v2_iq1_s+0x16d1>
    f605:	8b 44 24 88          	mov    eax,DWORD PTR [rsp-0x78]
    f609:	25 f8 ff ff 7f       	and    eax,0x7ffffff8
    f60e:	41 01 c6             	add    r14d,eax
    f611:	8b 84 24 08 02 00 00 	mov    eax,DWORD PTR [rsp+0x208]
    f618:	44 29 f0             	sub    eax,r14d
    f61b:	89 44 24 10          	mov    DWORD PTR [rsp+0x10],eax
    f61f:	83 f8 04             	cmp    eax,0x4
    f622:	0f 8c c2 02 00 00    	jl     f8ea <glm_vx_packed_batch_v2_iq1_s+0x19aa>
    f628:	48 8b 7c 24 e8       	mov    rdi,QWORD PTR [rsp-0x18]
    f62d:	85 ff                	test   edi,edi
    f62f:	0f 8e bc 02 00 00    	jle    f8f1 <glm_vx_packed_batch_v2_iq1_s+0x19b1>
    f635:	8b 44 24 84          	mov    eax,DWORD PTR [rsp-0x7c]
    f639:	c1 6c 24 10 02       	shr    DWORD PTR [rsp+0x10],0x2
    f63e:	44 89 f1             	mov    ecx,r14d
    f641:	41 bb 03 02 00 00    	mov    r11d,0x203
    f647:	41 bf 04 03 00 00    	mov    r15d,0x304
    f64d:	83 f8 01             	cmp    eax,0x1
    f650:	83 d0 00             	adc    eax,0x0
    f653:	48 89 44 24 c0       	mov    QWORD PTR [rsp-0x40],rax
    f658:	89 f8                	mov    eax,edi
    f65a:	48 89 44 24 e0       	mov    QWORD PTR [rsp-0x20],rax
    f65f:	8b 84 24 00 02 00 00 	mov    eax,DWORD PTR [rsp+0x200]
    f666:	44 8d 0c 85 00 00 00 	lea    r9d,[rax*4+0x0]
    f66d:	00 
    f66e:	0f af c8             	imul   ecx,eax
    f671:	44 89 4c 24 18       	mov    DWORD PTR [rsp+0x18],r9d
    f676:	49 b9 40 e4 ff ff ff 	movabs r9,0xffffffffffffe440
    f67d:	ff ff ff 
    f680:	4c 03 4c 24 08       	add    r9,QWORD PTR [rsp+0x8]
    f685:	89 4c 24 90          	mov    DWORD PTR [rsp-0x70],ecx
    f689:	31 c9                	xor    ecx,ecx
    f68b:	0f 1f 44 00 00       	nop    DWORD PTR [rax+rax*1+0x0]
    f690:	44 89 f0             	mov    eax,r14d
    f693:	0f af c7             	imul   eax,edi
    f696:	48 89 4c 24 f0       	mov    QWORD PTR [rsp-0x10],rcx
    f69b:	4c 89 74 24 a0       	mov    QWORD PTR [rsp-0x60],r14
    f6a0:	48 89 44 24 98       	mov    QWORD PTR [rsp-0x68],rax
    f6a5:	41 8d 46 01          	lea    eax,[r14+0x1]
    f6a9:	0f af c7             	imul   eax,edi
    f6ac:	48 89 44 24 b0       	mov    QWORD PTR [rsp-0x50],rax
    f6b1:	41 8d 46 02          	lea    eax,[r14+0x2]
    f6b5:	0f af c7             	imul   eax,edi
    f6b8:	48 89 44 24 88       	mov    QWORD PTR [rsp-0x78],rax
    f6bd:	41 8d 46 03          	lea    eax,[r14+0x3]
    f6c1:	0f af c7             	imul   eax,edi
    f6c4:	31 ff                	xor    edi,edi
    f6c6:	48 89 44 24 a8       	mov    QWORD PTR [rsp-0x58],rax
    f6cb:	0f 1f 44 00 00       	nop    DWORD PTR [rax+rax*1+0x0]
    f6d0:	8b 6c 24 90          	mov    ebp,DWORD PTR [rsp-0x70]
    f6d4:	8b 44 24 84          	mov    eax,DWORD PTR [rsp-0x7c]
    f6d8:	0f af c7             	imul   eax,edi
    f6db:	48 89 7c 24 b8       	mov    QWORD PTR [rsp-0x48],rdi
    f6e0:	c5 f9 ef c0          	vpxor  xmm0,xmm0,xmm0
    f6e4:	31 c9                	xor    ecx,ecx
    f6e6:	48 89 44 24 c8       	mov    QWORD PTR [rsp-0x38],rax
    f6eb:	0f 1f 44 00 00       	nop    DWORD PTR [rax+rax*1+0x0]
    f6f0:	48 8b 44 24 c8       	mov    rax,QWORD PTR [rsp-0x38]
    f6f5:	48 89 4c 24 d0       	mov    QWORD PTR [rsp-0x30],rcx
    f6fa:	89 6c 24 d8          	mov    DWORD PTR [rsp-0x28],ebp
    f6fe:	01 c1                	add    ecx,eax
    f700:	6b f9 32             	imul   edi,ecx,0x32
    f703:	48 63 cf             	movsxd rcx,edi
    f706:	8d 5f 22             	lea    ebx,[rdi+0x22]
    f709:	83 c7 02             	add    edi,0x2
    f70c:	44 0f b6 14 0e       	movzx  r10d,BYTE PTR [rsi+rcx*1]
    f711:	0f b6 4c 0e 01       	movzx  ecx,BYTE PTR [rsi+rcx*1+0x1]
    f716:	c1 e1 0a             	shl    ecx,0xa
    f719:	48 03 0c 24          	add    rcx,QWORD PTR [rsp]
    f71d:	c4 a1 7a 10 0c 91    	vmovss xmm1,DWORD PTR [rcx+r10*4]
    f723:	89 e9                	mov    ecx,ebp
    f725:	31 ed                	xor    ebp,ebp
    f727:	66 0f 1f 84 00 00 00 	nop    WORD PTR [rax+rax*1+0x0]
    f72e:	00 00 
    f730:	41 89 ea             	mov    r10d,ebp
    f733:	41 c1 ea 05          	shr    r10d,0x5
    f737:	c4 62 20 f7 e5       	bextr  r12d,ebp,r11d
    f73c:	48 63 c9             	movsxd rcx,ecx
    f73f:	46 8d 1c 53          	lea    r11d,[rbx+r10*2]
    f743:	46 8d 14 97          	lea    r10d,[rdi+r10*4]
    f747:	4d 63 db             	movsxd r11,r11d
    f74a:	46 0f b6 74 1e 01    	movzx  r14d,BYTE PTR [rsi+r11*1+0x1]
    f750:	42 0f b6 04 1e       	movzx  eax,BYTE PTR [rsi+r11*1]
    f755:	45 31 db             	xor    r11d,r11d
    f758:	45 84 f6             	test   r14b,r14b
    f75b:	45 89 f5             	mov    r13d,r14d
    f75e:	41 0f 99 c3          	setns  r11b
    f762:	45 01 e2             	add    r10d,r12d
    f765:	41 c1 e5 08          	shl    r13d,0x8
    f769:	4d 63 d2             	movsxd r10,r10d
    f76c:	41 09 c5             	or     r13d,eax
    f76f:	c4 c2 00 f7 c6       	bextr  eax,r14d,r15d
    f774:	47 8d 34 64          	lea    r14d,[r12+r12*2]
    f778:	c4 81 7a 10 24 99    	vmovss xmm4,DWORD PTR [r9+r11*4]
    f77e:	41 bb 03 02 00 00    	mov    r11d,0x203
    f784:	46 0f b6 14 16       	movzx  r10d,BYTE PTR [rsi+r10*1]
    f789:	8d 44 00 01          	lea    eax,[rax+rax*1+0x1]
    f78d:	c4 42 0b f7 f5       	shrx   r14d,r13d,r14d
    f792:	41 83 e6 07          	and    r14d,0x7
    f796:	c5 82 2a d0          	vcvtsi2ss xmm2,xmm15,eax
    f79a:	89 e8                	mov    eax,ebp
    f79c:	83 e0 06             	and    eax,0x6
    f79f:	41 c1 e6 08          	shl    r14d,0x8
    f7a3:	45 09 d6             	or     r14d,r10d
    f7a6:	c5 f2 59 d2          	vmulss xmm2,xmm1,xmm2
    f7aa:	42 8d 04 f0          	lea    eax,[rax+r14*8]
    f7ae:	41 0f b6 04 00       	movzx  eax,BYTE PTR [r8+rax*1]
    f7b3:	44 8d b0 00 ff ff ff 	lea    r14d,[rax-0x100]
    f7ba:	84 c0                	test   al,al
    f7bc:	44 0f 49 f0          	cmovns r14d,eax
    f7c0:	8d 45 01             	lea    eax,[rbp+0x1]
    f7c3:	c4 c1 02 2a de       	vcvtsi2ss xmm3,xmm15,r14d
    f7c8:	41 89 c6             	mov    r14d,eax
    f7cb:	83 e0 07             	and    eax,0x7
    f7ce:	41 c0 ee 03          	shr    r14b,0x3
    f7d2:	41 80 e6 03          	and    r14b,0x3
    f7d6:	45 0f b6 f6          	movzx  r14d,r14b
    f7da:	c5 da 58 db          	vaddss xmm3,xmm4,xmm3
    f7de:	47 8d 34 76          	lea    r14d,[r14+r14*2]
    f7e2:	c4 42 0b f7 f5       	shrx   r14d,r13d,r14d
    f7e7:	41 83 e6 07          	and    r14d,0x7
    f7eb:	41 c1 e6 08          	shl    r14d,0x8
    f7ef:	45 09 d6             	or     r14d,r10d
    f7f2:	42 8d 04 f0          	lea    eax,[rax+r14*8]
    f7f6:	41 0f b6 04 00       	movzx  eax,BYTE PTR [r8+rax*1]
    f7fb:	44 8d 90 00 ff ff ff 	lea    r10d,[rax-0x100]
    f802:	84 c0                	test   al,al
    f804:	44 0f 49 d0          	cmovns r10d,eax
    f808:	8d 41 04             	lea    eax,[rcx+0x4]
    f80b:	48 83 c5 02          	add    rbp,0x2
    f80f:	c4 c1 02 2a ea       	vcvtsi2ss xmm5,xmm15,r10d
    f814:	48 98                	cdqe   
    f816:	c5 da 58 ed          	vaddss xmm5,xmm4,xmm5
    f81a:	c5 ea 59 ed          	vmulss xmm5,xmm2,xmm5
    f81e:	c5 ea 59 d3          	vmulss xmm2,xmm2,xmm3
    f822:	c4 e2 79 18 d2       	vbroadcastss xmm2,xmm2
    f827:	c5 e8 59 14 8a       	vmulps xmm2,xmm2,XMMWORD PTR [rdx+rcx*4]
    f82c:	c4 e2 79 18 ed       	vbroadcastss xmm5,xmm5
    f831:	c5 d0 59 2c 82       	vmulps xmm5,xmm5,XMMWORD PTR [rdx+rax*4]
    f836:	83 c1 08             	add    ecx,0x8
    f839:	c5 f8 58 c2          	vaddps xmm0,xmm0,xmm2
    f83d:	c5 f8 58 c5          	vaddps xmm0,xmm0,xmm5
    f841:	48 81 fd 00 01 00 00 	cmp    rbp,0x100
    f848:	0f 85 e2 fe ff ff    	jne    f730 <glm_vx_packed_batch_v2_iq1_s+0x17f0>
    f84e:	48 8b 4c 24 d0       	mov    rcx,QWORD PTR [rsp-0x30]
    f853:	8b 6c 24 d8          	mov    ebp,DWORD PTR [rsp-0x28]
    f857:	48 ff c1             	inc    rcx
    f85a:	81 c5 00 04 00 00    	add    ebp,0x400
    f860:	48 3b 4c 24 c0       	cmp    rcx,QWORD PTR [rsp-0x40]
    f865:	0f 85 85 fe ff ff    	jne    f6f0 <glm_vx_packed_batch_v2_iq1_s+0x17b0>
    f86b:	48 8b 44 24 98       	mov    rax,QWORD PTR [rsp-0x68]
    f870:	48 8b 7c 24 b8       	mov    rdi,QWORD PTR [rsp-0x48]
    f875:	48 8b 4c 24 f8       	mov    rcx,QWORD PTR [rsp-0x8]
    f87a:	01 f8                	add    eax,edi
    f87c:	48 98                	cdqe   
    f87e:	c5 fa 11 04 81       	vmovss DWORD PTR [rcx+rax*4],xmm0
    f883:	48 8b 44 24 b0       	mov    rax,QWORD PTR [rsp-0x50]
    f888:	01 f8                	add    eax,edi
    f88a:	48 98                	cdqe   
    f88c:	c4 e3 79 17 04 81 01 	vextractps DWORD PTR [rcx+rax*4],xmm0,0x1
    f893:	48 8b 44 24 88       	mov    rax,QWORD PTR [rsp-0x78]
    f898:	01 f8                	add    eax,edi
    f89a:	48 98                	cdqe   
    f89c:	c4 e3 79 17 04 81 02 	vextractps DWORD PTR [rcx+rax*4],xmm0,0x2
    f8a3:	48 8b 44 24 a8       	mov    rax,QWORD PTR [rsp-0x58]
    f8a8:	01 f8                	add    eax,edi
    f8aa:	48 ff c7             	inc    rdi
    f8ad:	48 98                	cdqe   
    f8af:	c4 e3 79 17 04 81 03 	vextractps DWORD PTR [rcx+rax*4],xmm0,0x3
    f8b6:	48 3b 7c 24 e0       	cmp    rdi,QWORD PTR [rsp-0x20]
    f8bb:	0f 85 0f fe ff ff    	jne    f6d0 <glm_vx_packed_batch_v2_iq1_s+0x1790>
    f8c1:	8b 44 24 18          	mov    eax,DWORD PTR [rsp+0x18]
    f8c5:	4c 8b 74 24 a0       	mov    r14,QWORD PTR [rsp-0x60]
    f8ca:	48 8b 4c 24 f0       	mov    rcx,QWORD PTR [rsp-0x10]
    f8cf:	48 8b 7c 24 e8       	mov    rdi,QWORD PTR [rsp-0x18]
    f8d4:	01 44 24 90          	add    DWORD PTR [rsp-0x70],eax
    f8d8:	41 83 c6 04          	add    r14d,0x4
    f8dc:	ff c1                	inc    ecx
    f8de:	3b 4c 24 10          	cmp    ecx,DWORD PTR [rsp+0x10]
    f8e2:	0f 85 a8 fd ff ff    	jne    f690 <glm_vx_packed_batch_v2_iq1_s+0x1750>
    f8e8:	eb 13                	jmp    f8fd <glm_vx_packed_batch_v2_iq1_s+0x19bd>
    f8ea:	48 8b 7c 24 e8       	mov    rdi,QWORD PTR [rsp-0x18]
    f8ef:	eb 0c                	jmp    f8fd <glm_vx_packed_batch_v2_iq1_s+0x19bd>
    f8f1:	8b 44 24 10          	mov    eax,DWORD PTR [rsp+0x10]
    f8f5:	25 fc ff ff 7f       	and    eax,0x7ffffffc
    f8fa:	41 01 c6             	add    r14d,eax
    f8fd:	8b 84 24 08 02 00 00 	mov    eax,DWORD PTR [rsp+0x208]
    f904:	44 29 f0             	sub    eax,r14d
    f907:	89 44 24 e0          	mov    DWORD PTR [rsp-0x20],eax
    f90b:	83 f8 02             	cmp    eax,0x2
    f90e:	0f 8c 8b 02 00 00    	jl     fb9f <glm_vx_packed_batch_v2_iq1_s+0x1c5f>
    f914:	85 ff                	test   edi,edi
    f916:	0f 8e 77 02 00 00    	jle    fb93 <glm_vx_packed_batch_v2_iq1_s+0x1c53>
    f91c:	8b 44 24 84          	mov    eax,DWORD PTR [rsp-0x7c]
    f920:	d1 6c 24 e0          	shr    DWORD PTR [rsp-0x20],1
    f924:	49 b9 40 e4 ff ff ff 	movabs r9,0xffffffffffffe440
    f92b:	ff ff ff 
    f92e:	44 89 f1             	mov    ecx,r14d
    f931:	41 ba 03 02 00 00    	mov    r10d,0x203
    f937:	41 bf 04 03 00 00    	mov    r15d,0x304
    f93d:	83 f8 01             	cmp    eax,0x1
    f940:	83 d0 00             	adc    eax,0x0
    f943:	4c 03 4c 24 08       	add    r9,QWORD PTR [rsp+0x8]
    f948:	48 89 44 24 c0       	mov    QWORD PTR [rsp-0x40],rax
    f94d:	89 f8                	mov    eax,edi
    f94f:	48 89 44 24 88       	mov    QWORD PTR [rsp-0x78],rax
    f954:	8b 84 24 00 02 00 00 	mov    eax,DWORD PTR [rsp+0x200]
    f95b:	0f af c8             	imul   ecx,eax
    f95e:	01 c0                	add    eax,eax
    f960:	89 44 24 f0          	mov    DWORD PTR [rsp-0x10],eax
    f964:	89 4c 24 90          	mov    DWORD PTR [rsp-0x70],ecx
    f968:	31 c9                	xor    ecx,ecx
    f96a:	66 0f 1f 44 00 00    	nop    WORD PTR [rax+rax*1+0x0]
    f970:	44 89 f0             	mov    eax,r14d
    f973:	0f af c7             	imul   eax,edi
    f976:	48 89 4c 24 a8       	mov    QWORD PTR [rsp-0x58],rcx
    f97b:	4c 89 74 24 a0       	mov    QWORD PTR [rsp-0x60],r14
    f980:	48 89 44 24 98       	mov    QWORD PTR [rsp-0x68],rax
    f985:	41 8d 46 01          	lea    eax,[r14+0x1]
    f989:	0f af c7             	imul   eax,edi
    f98c:	31 ff                	xor    edi,edi
    f98e:	48 89 44 24 b0       	mov    QWORD PTR [rsp-0x50],rax
    f993:	66 66 66 66 2e 0f 1f 	data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
    f99a:	84 00 00 00 00 00 
    f9a0:	44 8b 5c 24 90       	mov    r11d,DWORD PTR [rsp-0x70]
    f9a5:	8b 44 24 84          	mov    eax,DWORD PTR [rsp-0x7c]
    f9a9:	0f af c7             	imul   eax,edi
    f9ac:	48 89 7c 24 b8       	mov    QWORD PTR [rsp-0x48],rdi
    f9b1:	c5 f9 ef c0          	vpxor  xmm0,xmm0,xmm0
    f9b5:	31 c9                	xor    ecx,ecx
    f9b7:	48 89 44 24 c8       	mov    QWORD PTR [rsp-0x38],rax
    f9bc:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]
    f9c0:	48 8b 44 24 c8       	mov    rax,QWORD PTR [rsp-0x38]
    f9c5:	48 89 4c 24 d0       	mov    QWORD PTR [rsp-0x30],rcx
    f9ca:	44 89 5c 24 d8       	mov    DWORD PTR [rsp-0x28],r11d
    f9cf:	01 c1                	add    ecx,eax
    f9d1:	44 6b f1 32          	imul   r14d,ecx,0x32
    f9d5:	49 63 ce             	movsxd rcx,r14d
    f9d8:	45 8d 66 22          	lea    r12d,[r14+0x22]
    f9dc:	41 83 c6 02          	add    r14d,0x2
    f9e0:	0f b6 3c 0e          	movzx  edi,BYTE PTR [rsi+rcx*1]
    f9e4:	0f b6 4c 0e 01       	movzx  ecx,BYTE PTR [rsi+rcx*1+0x1]
    f9e9:	c1 e1 0a             	shl    ecx,0xa
    f9ec:	48 03 0c 24          	add    rcx,QWORD PTR [rsp]
    f9f0:	c5 fa 10 0c b9       	vmovss xmm1,DWORD PTR [rcx+rdi*4]
    f9f5:	44 89 df             	mov    edi,r11d
    f9f8:	45 31 db             	xor    r11d,r11d
    f9fb:	0f 1f 44 00 00       	nop    DWORD PTR [rax+rax*1+0x0]
    fa00:	44 89 d9             	mov    ecx,r11d
    fa03:	c1 e9 05             	shr    ecx,0x5
    fa06:	c4 c2 28 f7 eb       	bextr  ebp,r11d,r10d
    fa0b:	31 db                	xor    ebx,ebx
    fa0d:	48 63 ff             	movsxd rdi,edi
    fa10:	45 8d 14 4c          	lea    r10d,[r12+rcx*2]
    fa14:	41 8d 0c 8e          	lea    ecx,[r14+rcx*4]
    fa18:	c5 fb 10 3c ba       	vmovsd xmm7,QWORD PTR [rdx+rdi*4]
    fa1d:	49 63 c2             	movsxd rax,r10d
    fa20:	44 0f b6 54 06 01    	movzx  r10d,BYTE PTR [rsi+rax*1+0x1]
    fa26:	0f b6 04 06          	movzx  eax,BYTE PTR [rsi+rax*1]
    fa2a:	45 84 d2             	test   r10b,r10b
    fa2d:	0f 99 c3             	setns  bl
    fa30:	c4 42 00 f7 ea       	bextr  r13d,r10d,r15d
    fa35:	41 c1 e2 08          	shl    r10d,0x8
    fa39:	01 e9                	add    ecx,ebp
    fa3b:	41 09 c2             	or     r10d,eax
    fa3e:	48 63 c1             	movsxd rax,ecx
    fa41:	c4 c1 7a 10 1c 99    	vmovss xmm3,DWORD PTR [r9+rbx*4]
    fa47:	44 89 db             	mov    ebx,r11d
    fa4a:	83 e3 06             	and    ebx,0x6
    fa4d:	47 8d 6c 2d 01       	lea    r13d,[r13+r13*1+0x1]
    fa52:	0f b6 0c 06          	movzx  ecx,BYTE PTR [rsi+rax*1]
    fa56:	8d 44 6d 00          	lea    eax,[rbp+rbp*2+0x0]
    fa5a:	c4 c1 02 2a d5       	vcvtsi2ss xmm2,xmm15,r13d
    fa5f:	c4 c2 7b f7 c2       	shrx   eax,r10d,eax
    fa64:	83 e0 07             	and    eax,0x7
    fa67:	c1 e0 08             	shl    eax,0x8
    fa6a:	c5 f2 59 d2          	vmulss xmm2,xmm1,xmm2
    fa6e:	09 c8                	or     eax,ecx
    fa70:	8d 04 c3             	lea    eax,[rbx+rax*8]
    fa73:	41 0f b6 04 00       	movzx  eax,BYTE PTR [r8+rax*1]
    fa78:	8d 98 00 ff ff ff    	lea    ebx,[rax-0x100]
    fa7e:	84 c0                	test   al,al
    fa80:	0f 49 d8             	cmovns ebx,eax
    fa83:	41 8d 43 01          	lea    eax,[r11+0x1]
    fa87:	89 c5                	mov    ebp,eax
    fa89:	c5 82 2a e3          	vcvtsi2ss xmm4,xmm15,ebx
    fa8d:	83 e0 07             	and    eax,0x7
    fa90:	40 c0 ed 03          	shr    bpl,0x3
    fa94:	40 80 e5 03          	and    bpl,0x3
    fa98:	40 0f b6 dd          	movzx  ebx,bpl
    fa9c:	c5 e2 58 e4          	vaddss xmm4,xmm3,xmm4
    faa0:	8d 1c 5b             	lea    ebx,[rbx+rbx*2]
    faa3:	c4 42 63 f7 d2       	shrx   r10d,r10d,ebx
    faa8:	c5 ea 59 e4          	vmulss xmm4,xmm2,xmm4
    faac:	41 83 e2 07          	and    r10d,0x7
    fab0:	41 c1 e2 08          	shl    r10d,0x8
    fab4:	41 09 ca             	or     r10d,ecx
    fab7:	c4 e2 79 18 e4       	vbroadcastss xmm4,xmm4
    fabc:	42 8d 04 d0          	lea    eax,[rax+r10*8]
    fac0:	41 ba 03 02 00 00    	mov    r10d,0x203
    fac6:	c5 c0 59 e4          	vmulps xmm4,xmm7,xmm4
    faca:	41 0f b6 04 00       	movzx  eax,BYTE PTR [r8+rax*1]
    facf:	8d 88 00 ff ff ff    	lea    ecx,[rax-0x100]
    fad5:	84 c0                	test   al,al
    fad7:	0f 49 c8             	cmovns ecx,eax
    fada:	8d 47 02             	lea    eax,[rdi+0x2]
    fadd:	49 83 c3 02          	add    r11,0x2
    fae1:	83 c7 04             	add    edi,0x4
    fae4:	c5 82 2a e9          	vcvtsi2ss xmm5,xmm15,ecx
    fae8:	48 98                	cdqe   
    faea:	c5 f8 58 c4          	vaddps xmm0,xmm0,xmm4
    faee:	c5 fb 10 34 82       	vmovsd xmm6,QWORD PTR [rdx+rax*4]
    faf3:	c5 e2 58 dd          	vaddss xmm3,xmm3,xmm5
    faf7:	c5 ea 59 d3          	vmulss xmm2,xmm2,xmm3
    fafb:	c4 e2 79 18 d2       	vbroadcastss xmm2,xmm2
    fb00:	c5 c8 59 d2          	vmulps xmm2,xmm6,xmm2
    fb04:	c5 f8 58 c2          	vaddps xmm0,xmm0,xmm2
    fb08:	49 81 fb 00 01 00 00 	cmp    r11,0x100
    fb0f:	0f 85 eb fe ff ff    	jne    fa00 <glm_vx_packed_batch_v2_iq1_s+0x1ac0>
    fb15:	48 8b 4c 24 d0       	mov    rcx,QWORD PTR [rsp-0x30]
    fb1a:	44 8b 5c 24 d8       	mov    r11d,DWORD PTR [rsp-0x28]
    fb1f:	48 ff c1             	inc    rcx
    fb22:	41 81 c3 00 02 00 00 	add    r11d,0x200
    fb29:	48 3b 4c 24 c0       	cmp    rcx,QWORD PTR [rsp-0x40]
    fb2e:	0f 85 8c fe ff ff    	jne    f9c0 <glm_vx_packed_batch_v2_iq1_s+0x1a80>
    fb34:	48 8b 44 24 98       	mov    rax,QWORD PTR [rsp-0x68]
    fb39:	48 8b 7c 24 b8       	mov    rdi,QWORD PTR [rsp-0x48]
    fb3e:	48 8b 4c 24 f8       	mov    rcx,QWORD PTR [rsp-0x8]
    fb43:	01 f8                	add    eax,edi
    fb45:	48 98                	cdqe   
    fb47:	c5 fa 11 04 81       	vmovss DWORD PTR [rcx+rax*4],xmm0
    fb4c:	48 8b 44 24 b0       	mov    rax,QWORD PTR [rsp-0x50]
    fb51:	01 f8                	add    eax,edi
    fb53:	48 ff c7             	inc    rdi
    fb56:	48 98                	cdqe   
    fb58:	c4 e3 79 17 04 81 01 	vextractps DWORD PTR [rcx+rax*4],xmm0,0x1
    fb5f:	48 3b 7c 24 88       	cmp    rdi,QWORD PTR [rsp-0x78]
    fb64:	0f 85 36 fe ff ff    	jne    f9a0 <glm_vx_packed_batch_v2_iq1_s+0x1a60>
    fb6a:	8b 44 24 f0          	mov    eax,DWORD PTR [rsp-0x10]
    fb6e:	4c 8b 74 24 a0       	mov    r14,QWORD PTR [rsp-0x60]
    fb73:	48 8b 4c 24 a8       	mov    rcx,QWORD PTR [rsp-0x58]
    fb78:	48 8b 7c 24 e8       	mov    rdi,QWORD PTR [rsp-0x18]
    fb7d:	01 44 24 90          	add    DWORD PTR [rsp-0x70],eax
    fb81:	41 83 c6 02          	add    r14d,0x2
    fb85:	ff c1                	inc    ecx
    fb87:	3b 4c 24 e0          	cmp    ecx,DWORD PTR [rsp-0x20]
    fb8b:	0f 85 df fd ff ff    	jne    f970 <glm_vx_packed_batch_v2_iq1_s+0x1a30>
    fb91:	eb 0c                	jmp    fb9f <glm_vx_packed_batch_v2_iq1_s+0x1c5f>
    fb93:	8b 44 24 e0          	mov    eax,DWORD PTR [rsp-0x20]
    fb97:	25 fe ff ff 7f       	and    eax,0x7ffffffe
    fb9c:	41 01 c6             	add    r14d,eax
    fb9f:	44 8b 8c 24 08 02 00 	mov    r9d,DWORD PTR [rsp+0x208]
    fba6:	00 
    fba7:	45 29 f1             	sub    r9d,r14d
    fbaa:	45 85 c9             	test   r9d,r9d
    fbad:	0f 9e c0             	setle  al
    fbb0:	85 ff                	test   edi,edi
    fbb2:	0f 9e c1             	setle  cl
    fbb5:	08 c1                	or     cl,al
    fbb7:	0f 85 66 02 00 00    	jne    fe23 <glm_vx_packed_batch_v2_iq1_s+0x1ee3>
    fbbd:	8b 44 24 84          	mov    eax,DWORD PTR [rsp-0x7c]
    fbc1:	4c 89 44 24 c0       	mov    QWORD PTR [rsp-0x40],r8
    fbc6:	4c 8b 64 24 08       	mov    r12,QWORD PTR [rsp+0x8]
    fbcb:	44 89 f1             	mov    ecx,r14d
    fbce:	41 bb 03 02 00 00    	mov    r11d,0x203
    fbd4:	44 89 4c 24 e0       	mov    DWORD PTR [rsp-0x20],r9d
    fbd9:	4c 8b 44 24 c0       	mov    r8,QWORD PTR [rsp-0x40]
    fbde:	83 f8 01             	cmp    eax,0x1
    fbe1:	83 d0 00             	adc    eax,0x0
    fbe4:	48 89 44 24 c8       	mov    QWORD PTR [rsp-0x38],rax
    fbe9:	89 f8                	mov    eax,edi
    fbeb:	48 89 44 24 88       	mov    QWORD PTR [rsp-0x78],rax
    fbf0:	8b 84 24 00 02 00 00 	mov    eax,DWORD PTR [rsp+0x200]
    fbf7:	0f af c8             	imul   ecx,eax
    fbfa:	89 c0                	mov    eax,eax
    fbfc:	48 89 44 24 f0       	mov    QWORD PTR [rsp-0x10],rax
    fc01:	48 b8 40 e4 ff ff ff 	movabs rax,0xffffffffffffe440
    fc08:	ff ff ff 
    fc0b:	49 01 c4             	add    r12,rax
    fc0e:	48 89 4c 24 90       	mov    QWORD PTR [rsp-0x70],rcx
    fc13:	ff c1                	inc    ecx
    fc15:	48 89 4c 24 98       	mov    QWORD PTR [rsp-0x68],rcx
    fc1a:	31 c9                	xor    ecx,ecx
    fc1c:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]
    fc20:	4c 89 74 24 a0       	mov    QWORD PTR [rsp-0x60],r14
    fc25:	44 0f af f7          	imul   r14d,edi
    fc29:	31 ff                	xor    edi,edi
    fc2b:	48 89 4c 24 a8       	mov    QWORD PTR [rsp-0x58],rcx
    fc30:	4c 89 74 24 b0       	mov    QWORD PTR [rsp-0x50],r14
    fc35:	66 66 2e 0f 1f 84 00 	data16 cs nop WORD PTR [rax+rax*1+0x0]
    fc3c:	00 00 00 00 
    fc40:	48 8b 5c 24 90       	mov    rbx,QWORD PTR [rsp-0x70]
    fc45:	4c 8b 74 24 98       	mov    r14,QWORD PTR [rsp-0x68]
    fc4a:	8b 44 24 84          	mov    eax,DWORD PTR [rsp-0x7c]
    fc4e:	0f af c7             	imul   eax,edi
    fc51:	48 89 7c 24 b8       	mov    QWORD PTR [rsp-0x48],rdi
    fc56:	c5 f9 ef c0          	vpxor  xmm0,xmm0,xmm0
    fc5a:	31 c9                	xor    ecx,ecx
    fc5c:	48 89 44 24 d0       	mov    QWORD PTR [rsp-0x30],rax
    fc61:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
    fc68:	0f 1f 84 00 00 00 00 
    fc6f:	00 
    fc70:	48 8b 44 24 d0       	mov    rax,QWORD PTR [rsp-0x30]
    fc75:	48 89 4c 24 d8       	mov    QWORD PTR [rsp-0x28],rcx
    fc7a:	45 31 ed             	xor    r13d,r13d
    fc7d:	01 c8                	add    eax,ecx
    fc7f:	6b c8 32             	imul   ecx,eax,0x32
    fc82:	48 63 c1             	movsxd rax,ecx
    fc85:	8d 79 22             	lea    edi,[rcx+0x22]
    fc88:	83 c1 02             	add    ecx,0x2
    fc8b:	44 0f b6 0c 06       	movzx  r9d,BYTE PTR [rsi+rax*1]
    fc90:	0f b6 44 06 01       	movzx  eax,BYTE PTR [rsi+rax*1+0x1]
    fc95:	c1 e0 0a             	shl    eax,0xa
    fc98:	48 03 04 24          	add    rax,QWORD PTR [rsp]
    fc9c:	c4 a1 7a 10 0c 88    	vmovss xmm1,DWORD PTR [rax+r9*4]
    fca2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
    fca9:	1f 84 00 00 00 00 00 
    fcb0:	45 89 ef             	mov    r15d,r13d
    fcb3:	41 c1 ef 05          	shr    r15d,0x5
    fcb7:	c4 42 20 f7 d5       	bextr  r10d,r13d,r11d
    fcbc:	42 8d 04 7f          	lea    eax,[rdi+r15*2]
    fcc0:	48 98                	cdqe   
    fcc2:	0f b6 6c 06 01       	movzx  ebp,BYTE PTR [rsi+rax*1+0x1]
    fcc7:	44 0f b6 0c 06       	movzx  r9d,BYTE PTR [rsi+rax*1]
    fccc:	31 c0                	xor    eax,eax
    fcce:	40 84 ed             	test   bpl,bpl
    fcd1:	41 89 eb             	mov    r11d,ebp
    fcd4:	0f 99 c0             	setns  al
    fcd7:	41 c1 e3 08          	shl    r11d,0x8
    fcdb:	45 09 cb             	or     r11d,r9d
    fcde:	41 b9 04 03 00 00    	mov    r9d,0x304
    fce4:	c4 c1 7a 10 24 84    	vmovss xmm4,DWORD PTR [r12+rax*4]
    fcea:	42 8d 04 2b          	lea    eax,[rbx+r13*1]
    fcee:	c4 62 30 f7 cd       	bextr  r9d,ebp,r9d
    fcf3:	42 8d 2c b9          	lea    ebp,[rcx+r15*4]
    fcf7:	48 98                	cdqe   
    fcf9:	47 8d 4c 09 01       	lea    r9d,[r9+r9*1+0x1]
    fcfe:	44 01 d5             	add    ebp,r10d
    fd01:	47 8d 14 52          	lea    r10d,[r10+r10*2]
    fd05:	4c 63 fd             	movsxd r15,ebp
    fd08:	42 0f b6 2c 3e       	movzx  ebp,BYTE PTR [rsi+r15*1]
    fd0d:	c4 42 2b f7 d3       	shrx   r10d,r11d,r10d
    fd12:	45 89 eb             	mov    r11d,r13d
    fd15:	41 83 e3 06          	and    r11d,0x6
    fd19:	41 83 e2 07          	and    r10d,0x7
    fd1d:	41 c1 e2 08          	shl    r10d,0x8
    fd21:	41 09 ea             	or     r10d,ebp
    fd24:	47 8d 1c d3          	lea    r11d,[r11+r10*8]
    fd28:	47 0f b6 1c 18       	movzx  r11d,BYTE PTR [r8+r11*1]
    fd2d:	41 8d ab 00 ff ff ff 	lea    ebp,[r11-0x100]
    fd34:	45 84 db             	test   r11b,r11b
    fd37:	41 0f 49 eb          	cmovns ebp,r11d
    fd3b:	45 8d 5d 01          	lea    r11d,[r13+0x1]
    fd3f:	41 83 e3 07          	and    r11d,0x7
    fd43:	c5 82 2a d5          	vcvtsi2ss xmm2,xmm15,ebp
    fd47:	47 8d 14 d3          	lea    r10d,[r11+r10*8]
    fd4b:	47 0f b6 14 10       	movzx  r10d,BYTE PTR [r8+r10*1]
    fd50:	c5 da 58 d2          	vaddss xmm2,xmm4,xmm2
    fd54:	45 8d 9a 00 ff ff ff 	lea    r11d,[r10-0x100]
    fd5b:	45 84 d2             	test   r10b,r10b
    fd5e:	45 0f 49 da          	cmovns r11d,r10d
    fd62:	c4 c1 02 2a db       	vcvtsi2ss xmm3,xmm15,r11d
    fd67:	41 bb 03 02 00 00    	mov    r11d,0x203
    fd6d:	c5 da 58 db          	vaddss xmm3,xmm4,xmm3
    fd71:	c4 c1 02 2a e1       	vcvtsi2ss xmm4,xmm15,r9d
    fd76:	c5 f2 59 e4          	vmulss xmm4,xmm1,xmm4
    fd7a:	c5 da 59 d2          	vmulss xmm2,xmm4,xmm2
    fd7e:	c5 ea 59 14 82       	vmulss xmm2,xmm2,DWORD PTR [rdx+rax*4]
    fd83:	43 8d 04 2e          	lea    eax,[r14+r13*1]
    fd87:	c5 da 59 db          	vmulss xmm3,xmm4,xmm3
    fd8b:	49 83 c5 02          	add    r13,0x2
    fd8f:	48 98                	cdqe   
    fd91:	c5 e2 59 1c 82       	vmulss xmm3,xmm3,DWORD PTR [rdx+rax*4]
    fd96:	c5 fa 58 c2          	vaddss xmm0,xmm0,xmm2
    fd9a:	c5 fa 58 c3          	vaddss xmm0,xmm0,xmm3
    fd9e:	49 81 fd 00 01 00 00 	cmp    r13,0x100
    fda5:	0f 85 05 ff ff ff    	jne    fcb0 <glm_vx_packed_batch_v2_iq1_s+0x1d70>
    fdab:	48 8b 4c 24 d8       	mov    rcx,QWORD PTR [rsp-0x28]
    fdb0:	49 81 c6 00 01 00 00 	add    r14,0x100
    fdb7:	48 81 c3 00 01 00 00 	add    rbx,0x100
    fdbe:	48 ff c1             	inc    rcx
    fdc1:	48 3b 4c 24 c8       	cmp    rcx,QWORD PTR [rsp-0x38]
    fdc6:	0f 85 a4 fe ff ff    	jne    fc70 <glm_vx_packed_batch_v2_iq1_s+0x1d30>
    fdcc:	48 8b 44 24 b0       	mov    rax,QWORD PTR [rsp-0x50]
    fdd1:	48 8b 7c 24 b8       	mov    rdi,QWORD PTR [rsp-0x48]
    fdd6:	48 8b 4c 24 f8       	mov    rcx,QWORD PTR [rsp-0x8]
    fddb:	01 f8                	add    eax,edi
    fddd:	48 ff c7             	inc    rdi
    fde0:	48 98                	cdqe   
    fde2:	c5 fa 11 04 81       	vmovss DWORD PTR [rcx+rax*4],xmm0
    fde7:	48 3b 7c 24 88       	cmp    rdi,QWORD PTR [rsp-0x78]
    fdec:	0f 85 4e fe ff ff    	jne    fc40 <glm_vx_packed_batch_v2_iq1_s+0x1d00>
    fdf2:	48 8b 44 24 f0       	mov    rax,QWORD PTR [rsp-0x10]
    fdf7:	4c 8b 74 24 a0       	mov    r14,QWORD PTR [rsp-0x60]
    fdfc:	48 8b 4c 24 a8       	mov    rcx,QWORD PTR [rsp-0x58]
    fe01:	44 8b 4c 24 e0       	mov    r9d,DWORD PTR [rsp-0x20]
    fe06:	48 8b 7c 24 e8       	mov    rdi,QWORD PTR [rsp-0x18]
    fe0b:	48 01 44 24 98       	add    QWORD PTR [rsp-0x68],rax
    fe10:	48 01 44 24 90       	add    QWORD PTR [rsp-0x70],rax
    fe15:	41 ff c6             	inc    r14d
    fe18:	ff c1                	inc    ecx
    fe1a:	44 39 c9             	cmp    ecx,r9d
    fe1d:	0f 85 fd fd ff ff    	jne    fc20 <glm_vx_packed_batch_v2_iq1_s+0x1ce0>
    fe23:	31 c0                	xor    eax,eax
    fe25:	48 81 c4 c8 01 00 00 	add    rsp,0x1c8
    fe2c:	5b                   	pop    rbx
    fe2d:	41 5c                	pop    r12
    fe2f:	41 5d                	pop    r13
    fe31:	41 5e                	pop    r14
    fe33:	41 5f                	pop    r15
    fe35:	5d                   	pop    rbp
    fe36:	c5 f8 77             	vzeroupper 
    fe39:	c3                   	ret    

Disassembly of section .fini:
