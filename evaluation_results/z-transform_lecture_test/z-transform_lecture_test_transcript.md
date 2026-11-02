# Academic Lecture Transcript: Lecture 12 Z - Transform.wav

**Date:** 2026-11-02 15:37 | **Audio Duration:** 32:22 | **Engine:** Whisper-Small + Qwen2.5-1.5B (CUDA SM87)

---

**[00:00 - 00:10]** ♪♪♪

**[00:10 - 00:20]** The

**[00:20 - 00:25]** 

**[00:25 - 00:33]** So, today we will talk about the Z-transform. So, we have talked about the signals, we have talked about the systems.

**[00:33 - 00:36]** And they are discrete implementation we talk about.

**[00:36 - 00:39]** So, today we will talk about the Z-transform.

**[00:39 - 00:42]** Before we start with the Z-transform,

**[00:42 - 00:44]** You do not know what is task form.

**[00:44 - 00:49]** Why do we use transforms like the Z-transform? Are there other types of transforms used in engineering?

**[00:49 - 00:52]** There are lot of kinds of stuff. What is task form?

**[00:52 - 00:54]** Why you go for dead task movement?

**[00:54 - 00:57]** You have already learned about the Laplace transform.

**[00:57 - 01:00]** So, why these transforms are required?

**[01:00 - 01:01]** 

**[01:01 - 01:03]** So transform means

**[01:03 - 01:06]** Change of representation.

**[01:06 - 01:11]** So, suppose something is there, let us say

**[01:11 - 01:14]** This is very object.

**[01:14 - 01:17]** I can see this side of the object.

**[01:17 - 01:20]** But I cannot see this side of the object.

**[01:20 - 01:24]** Now, if I want to see this side of the object,

**[01:24 - 01:25]** What I have to do?

**[01:25 - 01:28]** I have to take it to rotate it.

**[01:28 - 01:32]** So, irrotation can be one kind of path one.

**[01:32 - 01:38]** So, I am representing the object in different planes.

**[01:38 - 01:41]** So in that case,

**[01:41 - 01:43]** That is called "Toss-Pong".

**[01:43 - 01:46]** So, let us talk about the Laplace transform.

**[01:46 - 01:49]** Why do I talk about the Laplace transform? What have we accomplished so far?

**[01:49 - 01:55]** Now let us transform any time domain system or time domain signal

**[01:55 - 01:59]** When I want to make a frequency domain, which is called S domain.

**[01:59 - 02:03]** So, I want to decompose, I want to transform,

**[02:03 - 02:07]** A time-domain signal, continuous-time domain signal

**[02:07 - 02:11]** To a another plane which is called S plane.

**[02:11 - 02:13]** Laplacian plane.

**[02:13 - 02:16]** That is the Laplace transform of the left-locked function.

**[02:16 - 02:19]** I am just calling

**[02:19 - 02:27]** The time domain representation to an another, so any vector, let's talk about the signal, signal is nothing but a vector.

**[02:27 - 02:29]** Let's talk about the history.

**[02:29 - 02:35]** So I want one representation of the vector is there, I want another representation.

**[02:35 - 02:38]** Time-doubling signal.

**[02:38 - 02:41]** It is there, I have a time-domain signal like this.

**[02:41 - 02:45]** Let us say I have a time-lover signal like this.

**[02:45 - 02:47]** But

**[02:47 - 02:51]** I want to represent this signal in another domain.

**[02:51 - 02:56]** This is a complex domain which is called S domain which is called Laplacian transfer.

**[02:56 - 02:57]** All right.

**[02:57 - 03:00]** What is complex domain? That means

**[03:00 - 03:06]** I am saying this time domain signal can be represented in a complex

**[03:06 - 03:10]** Coordinate, one is real and another is imaginary axis.

**[03:10 - 03:13]** Then I plot this signal in this.

**[03:13 - 03:15]** Representation.

**[03:15 - 03:17]** Then I call it the Laplace transform.

**[03:17 - 03:21]** So, for continuous signal,

**[03:21 - 03:22]** We use the Laplace transform.

**[03:22 - 03:27]** Now, why I have a digital or discrete signal?

**[03:27 - 03:34]** Or I can say it's a digital signal, but a discrete signal means that time is not continuous. Thus, $x[n]$.

**[03:34 - 03:36]** Here we go!

**[03:36 - 03:38]** Real transformation.

**[03:38 - 03:45]** What is $S$? $S$ is simply another symbol used for the complex frequency variable in the context of the Laplace transform applied to the continuous domain.

**[03:45 - 03:46]** Something

**[03:46 - 03:53]** So, that is nothing but a complex, so this is a, any S, S is nothing but a complex number.

**[03:53 - 03:57]** Similarly, Z is also a complex number.

**[03:57 - 04:00]** So, any discrete signal

**[04:00 - 04:07]** When I want to represent it in a z plane, so z plane again is a complex plane.

**[04:07 - 04:10]** I have a real axis, I just imagine a real axis.

**[04:10 - 04:13]** So, when I have a discrete signal,

**[04:13 - 04:17]** I want to represent in their domain.

**[04:17 - 04:20]** Which is equivalent to is called frequency domain.

**[04:20 - 04:23]** So time domain signal

**[04:23 - 04:26]** To frequency domain signal we call

**[04:26 - 04:27]** $J$-Class form.

**[04:27 - 04:28]** 

**[04:28 - 04:30]** Okay.

**[04:30 - 04:33]** So, let us say what is Z transfer?

**[04:33 - 04:42]** The Z transform of a discrete-time signal $x[n]$ is defined as a power series: this is the definition of Z transform.

**[04:42 - 04:45]** So what I am doing, I have an

**[04:45 - 04:48]** Digital signal which is $x[n]$

**[04:48 - 04:49]** 

**[04:49 - 04:56]** I want to transform it in Z plane. What is the formula for transformation?

**[04:56 - 05:01]** $n = -\infty \cdot x[n] \cdot z^{-1}$

**[05:01 - 05:02]** And

**[05:02 - 05:06]** So if m is positive,

**[05:06 - 05:14]** And so if $m > 0$, then it is $x[n] \cdot z^{-n}$. And if $n < 0$, then it is $x[n] \cdot z^{+n}$.

**[05:14 - 05:15]** Okay.

**[05:15 - 05:17]** So I am

**[05:17 - 05:19]** It's humble.

**[05:19 - 05:22]** So, x sub z is n enough for x sub z.

**[05:22 - 05:26]** So in z plane representation of xn,

**[05:26 - 05:31]** I go to the z plane, where z is a complex number.

**[05:31 - 05:32]** 

**[05:32 - 05:35]** And when I do that,

**[05:35 - 05:40]** The new signal which is represented in z plane is called x from z.

**[05:40 - 05:44]** Now, question is, why should I do that?

**[05:44 - 05:45]** 

**[05:45 - 05:52]** That we have already said that suppose I want to see this space of this object

**[05:52 - 05:58]** I have to rotate it. So rotation I have done because I want to see this phase.

**[05:58 - 05:59]** Pray.

**[05:59 - 06:02]** Now, why you would do for Jentak?

**[06:02 - 06:03]** All right.

**[06:03 - 06:05]** Why did we quit?

**[06:05 - 06:08]** Why do you want to pursue a career in left-luster?

**[06:08 - 06:12]** Why is it required? It is a cycle of mathematics.

**[06:12 - 06:13]** Nope.

**[06:13 - 06:15]** So what is that?

**[06:15 - 06:23]** The $Z$-transform is required when we want to analyze a discrete system.

**[06:23 - 06:25]** Which is called the "Pauli zero" concept.

**[06:25 - 06:31]** Similarly, the $Laplace$-transform is used when you want to analyze a system

**[06:31 - 06:33]** Which is called continuous system

**[06:33 - 06:39]** I go for the $L$-domain (last). Here also, if I want to analyze it.

**[06:39 - 06:43]** So, analysis will become easier.

**[06:43 - 06:47]** How about discrete system when I go to the Z domain?

**[06:47 - 06:48]** 

**[06:48 - 06:50]** 

**[06:50 - 06:58]** So, that is why I request, so this will be easy to analyze a discrete signum for discrete system.

**[06:58 - 07:03]** So, Z domain is called frequency domain representation.

**[07:03 - 07:05]** So that's why I went.

**[07:05 - 07:08]** I have to establish what kind of analysis

**[07:08 - 07:17]** So, at the end of the lecture you can understand why this get the sum required. If the get the sum is not there, what we cannot do?

**[07:17 - 07:18]** 

**[07:18 - 07:21]** So initially I said

**[07:21 - 07:29]** x sub n is the domain representation of x sub n. x sub n is a discrete signal

**[07:29 - 07:32]** $x(j) = \frac{j}{\sigma}$

**[07:32 - 07:33]** 

**[07:33 - 07:43]** And z is a complex variable. It has a real axis. It can be real and imaginary plane. So, z in any point

**[07:43 - 07:47]** Can be represented by $R \exp(j\omega)$.

**[07:47 - 07:48]** 

**[07:48 - 07:50]** 

**[07:50 - 07:54]** $z = re^{j\theta}$

**[07:54 - 07:58]** $e^{j\omega} = \cos(\omega)$

**[07:58 - 08:03]** $\theta$ is the phase angle of $z$.

**[08:03 - 08:06]** $j\omega$ represents a complex exponential with unit magnitude.

**[08:06 - 08:08]** $\omega$ denotes angular frequency in radians per second.

**[08:08 - 08:15]** This axis is positive, this side, this side, this side it is negative and what is anticlockwise it is called?

**[08:15 - 08:16]** Wonderful.

**[08:16 - 08:20]** So $-j\omega_B$.

**[08:20 - 08:26]** So, if $\omega > 0$, this side is $e^{-j\theta}$.

**[08:26 - 08:27]** 

**[08:27 - 08:31]** So, that is the requirement for

**[08:31 - 08:33]** Geg-Dohmen.

**[08:33 - 08:34]** 

**[08:34 - 08:36]** Okay?

**[08:36 - 08:37]** 

**[08:37 - 08:40]** Why do we use the z-transform?

**[08:40 - 08:44]** That means we want to represent a discrete signal

**[08:44 - 08:48]** In a complex field plane which is called Z,

**[08:48 - 08:49]** Doorman.

**[08:49 - 08:51]** Well,

**[08:51 - 08:58]** Is the Jetta, when the Jetta sum will be valid? I want to make it task form, but task form has to exist.

**[08:58 - 08:59]** 

**[08:59 - 09:04]** If you see, here it is an infinite power series.

**[09:04 - 09:14]** $x(z) = \sum_{n=-\infty}^{\infty} x[n]z^{-n}$

**[09:14 - 09:15]** $y(t)$ low-pass filter.

**[09:15 - 09:17]** So why do we exist?

**[09:17 - 09:20]** If x sub z is finite,

**[09:20 - 09:28]** Unless, how can I do it? If it is infinite value, then what is the presentation I can get?

**[09:28 - 09:29]** 

**[09:29 - 09:31]** So x did

**[09:31 - 09:36]** When we do task form, when xi is finite.

**[09:36 - 09:39]** That means this series, this term

**[09:39 - 09:41]** Has to be croned and

**[09:41 - 09:45]** finite number, that means it has to converge.

**[09:45 - 09:48]** Any tough quorum?

**[09:48 - 09:51]** When I want to do it, this mask has to be concurred.

**[09:51 - 09:55]** Unless if it is infinite, I cannot see anything there.

**[09:55 - 09:56]** 

**[09:56 - 10:00]** So, for which Z value?

**[10:00 - 10:06]** This xz has a finite value or the transform is or series is converge.

**[10:06 - 10:07]** That is called

**[10:07 - 10:10]** Reason of convergence.

**[10:10 - 10:12]** ROC.

**[10:12 - 10:15]** Reversion of convergence.

**[10:15 - 10:20]** For both $z$-values, so $z$-value is nothing but the $z$-plane.

**[10:20 - 10:25]** Good xy plane, real part and every jet has a real part and imaginary part.

**[10:25 - 10:27]** So, there is a z plane.

**[10:27 - 10:31]** So, I transform that discrete signal into the jet plane

**[10:31 - 10:35]** But that transform only useful when that

**[10:35 - 10:39]** xz has a finite value or I can say that

**[10:39 - 10:41]** For the Bohr's general rule,

**[10:41 - 10:44]** For which x is quite nice?

**[10:44 - 10:52]** Those $z$-values are called regions of convergence in the $z$-plane.

**[10:52 - 10:53]** 

**[10:53 - 10:54]** 

**[10:54 - 10:57]** Honestly.

**[10:57 - 10:58]** 

**[10:58 - 11:07]** So, when I perform a z-transform, you should consider why this transformation was chosen.

**[11:07 - 11:11]** Convergence. Now I come to the concept of the Z. What is Z?

**[11:11 - 11:12]** 

**[11:12 - 11:20]** $z^{-n}x(z) = -\infty$.

**[11:20 - 11:30]** $n = -\infty$ to $+\infty$, then $x[n]z^{-n}$. So, if we either reach $z^{-n}$ or $+ \infty$, then $x[n]$ is equal to zero.

**[11:30 - 11:39]** $z^n$ when $n > 0$, then $z^{-n}$ when $n < 0$.

**[11:39 - 11:41]** So what is it?

**[11:41 - 11:47]** I said z is nothing but a plane, z plane, so I have an xy.

**[11:47 - 11:52]** What they had told is two hallows. One is real hallow, another is imaginary hallow.

**[11:52 - 11:56]** So, any, so suppose I have a real axis and imaginary axis.

**[11:56 - 11:59]** This point is $a + j\omega$.

**[11:59 - 12:04]** I can represent this also by an $R \angle \theta$.

**[12:04 - 12:05]** 

**[12:05 - 12:12]** So, I can say instead of $A + iB$, I can say it is $R \angle \theta$.

**[12:12 - 12:13]** See guys the angle!

**[12:13 - 12:14]** 

**[12:14 - 12:22]** So, $z^{-n}$ can be represented as $r^{e^{\theta}}$ where $\theta = \frac{2\pi}{N}$.

**[12:22 - 12:24]** Deal with me, girl.

**[12:24 - 12:34]** When I say $z^{-n}$, where $r^{e^{\theta}}$, let's say $r^{e^{j\omega}}$.

**[12:34 - 12:38]** I can say $-jn\omega$.

**[12:38 - 12:41]** Now, when the n is positive,

**[12:41 - 12:44]** I can say $r^n e^{j\omega n}$.

**[12:44 - 12:54]** So, $e^{j\omega n}$, which is nothing but $e^{-j\theta} = \cos{\theta} - j\sin{\theta}$.

**[12:54 - 12:58]** Similarly, $e^{+j\theta}$ is $\cos{\theta} + j\sin{\theta}$.

**[12:58 - 13:00]** Different people.

**[13:00 - 13:01]** 

**[13:01 - 13:05]** So, Z has a real value and has an

**[13:05 - 13:15]** So, $A + iB$ where $R = \frac{1}{2}(\text{Re}(A) - \text{Im}(B))$, $\omega = \tan^{-1}\left(\frac{\text{Im}(A)}{\text{Re}(B)}\right)$

**[13:15 - 13:19]** So, the first part is the integral of the integral of the integral of the integral

**[13:19 - 13:21]** $\omega_n$.

**[13:21 - 13:24]** If I want to represent

**[13:24 - 13:34]** In a polar form, we can say $r \cdot e^{-j\omega_n}$, where $z = z^n$, and when $z = z^m$.

**[13:34 - 13:37]** I can say this is positive.

**[13:37 - 13:38]** 

**[13:38 - 13:41]** So what is Ahm?

**[13:41 - 13:44]** , R will determine

**[13:44 - 13:51]** So, what is Z? Z is nothing but a Gaussian, who may a Gaussian sign? So, Gaussian sign is nothing but an oscillation.

**[13:51 - 13:52]** 

**[13:52 - 13:56]** Physically, there is nothing but an oscillation.

**[13:56 - 14:00]** $e^{j\omega t}$, $P(t)$, plus $j \sin(\omega t)$.

**[14:00 - 14:01]** 

**[14:01 - 14:05]** An oscillation has an amplitude and has an

**[14:05 - 14:13]** The amplitude and \omega (the angular frequency) determine the frequency or the amplitude of the

**[14:13 - 14:14]** 

**[14:14 - 14:16]** 

**[14:16 - 14:17]** 

**[14:17 - 14:27]** So, frequency is determined by $\omega$ and $R$. Thus, I can say that for every $Z$, there is an oscillation with an amplitude of $A = \frac{1}{\sqrt{\pi R}}e^{-\frac{\omega^2}{4R}}$.

**[14:27 - 14:32]** The frequency depends on R and the frequency will depends on

**[14:32 - 14:33]** Ormenga!

**[14:33 - 14:34]** 

**[14:34 - 14:43]** So, whether it will be decayed, how much it will be decayed that depends on the value of R.

**[14:43 - 14:44]** 

**[14:44 - 14:50]** Now all of you have heard about the unit circle in the Laplace transform.

**[14:50 - 14:53]** What is the concept of unit function?

**[14:53 - 14:59]** So that's why I said "plot of $z^n$", let's see, what if I want to plot

**[14:59 - 15:02]** So, this axis is my real axis.

**[15:02 - 15:05]** This axis is my imaginary axis.

**[15:05 - 15:08]** So $y$-axis is the imaginary.

**[15:08 - 15:10]** And $x$-axis is the real axis.

**[15:10 - 15:11]** 

**[15:11 - 15:15]** Now, the stability

**[15:15 - 15:18]** The system is stable if it is within an unit circle.

**[15:18 - 15:19]** 

**[15:19 - 15:28]** Now, if you see the jet value, this blue circle is the unit circle. The meaning is $r = 1$.

**[15:28 - 15:29]** 

**[15:29 - 15:36]** When $r = 1$, $z^n$ equals $r^n$.

**[15:36 - 15:40]** It would be for $\omega \rightarrow -j\omega$ or $\omega \rightarrow +j\omega$.

**[15:40 - 15:42]** Gaining is wonderful.

**[15:42 - 15:43]** 

**[15:43 - 15:53]** When $r = 1$, we can say that it is simply equal to $e^{j\omega n}$. This represents a constant oscillatory behavior.

**[15:53 - 15:57]** So, when the Z value, when the Z value will be here.

**[15:57 - 16:01]** This is a $z$-plane; this entire plane represents the $z$-plane.

**[16:01 - 16:07]** So, when the $Z$ value is here, where the $R$ value equals 1, $R = 1$.

**[16:07 - 16:10]** Then I said any point on this unit circle

**[16:10 - 16:16]** It is nothing but $e^{j\omega n}$. This different point means different $\omega$.

**[16:16 - 16:20]** But, so that is constant oscillation.

**[16:20 - 16:26]** So, I can get a constant oscillation of different frequencies.

**[16:26 - 16:28]** If I go for the, to the unit side.

**[16:28 - 16:32]** But inside the unit circle that means

**[16:32 - 16:35]** $R < 1$.

**[16:35 - 16:38]** What is the meaning of inside the unit circle?

**[16:38 - 16:40]** $a \leq 1$.

**[16:40 - 16:44]** $R < 1$, where $R$ represents the resistance value in ohms (Ω).

**[16:44 - 16:52]** This implies that the oscillation has not decayed since the series $\sum_{n=0}^{\infty}$ converges.

**[16:52 - 16:57]** $r^n$ for $r < 1$.

**[16:57 - 17:02]** So, I can say it will be decayed down, so the oscillation will be decayed down.

**[17:02 - 17:03]** 

**[17:03 - 17:12]** Now, if I say outside the English circle, that means r is greater than 1, here, here, here, here, here,

**[17:12 - 17:16]** So when I say $R > 1$,

**[17:16 - 17:26]** Then I can say $\sum_{n=0}^{\infty} r^n$ converges to infinity, so there is an infinite gain of $G$.

**[17:26 - 17:30]** So, oscillation is increasing.

**[17:30 - 17:35]** The amplitude of the oscillation is increasing if n goes to infinite.

**[17:35 - 17:39]** Then it will be infinite series. If it is infinite series, the Rallo will be infinite.

**[17:39 - 17:41]** 

**[17:41 - 17:42]** 

**[17:42 - 17:47]** So, depending on the value of R,

**[17:47 - 17:51]** Z, where you are present in Z plane?

**[17:51 - 17:53]** Now, oscillation will be different.

**[17:53 - 17:55]** So what is the stimulus system?

**[17:55 - 17:58]** What is the concept of unit circle?

**[17:58 - 18:00]** Unit circle means

**[18:00 - 18:10]** The concept of a unit circle refers to any point that lies on or inside its boundary; points outside this circle are considered "outside" and may undergo different behaviors, including potential oscillation or instability.

**[18:10 - 18:12]** The system is stable.

**[18:12 - 18:22]** Unless what will happen? Outside the unit circle, it creates an infinite amplitude oscillation and goes, I do not know, there is no control.

**[18:22 - 18:32]** So, that is the concept of unit circle. That is exist in S plane, that is exist in Z plane, both the

**[18:32 - 18:33]** 

**[18:33 - 18:34]** 

**[18:34 - 18:37]** Is it clear?

**[18:37 - 18:45]** Some examples in z-domain. Let's say we have a signal $x[n]$ with initial condition $x[0] = 0$.

**[18:45 - 18:48]** This arrow indicates the initial condition, which is zero for this signal.

**[18:48 - 18:51]** Therefore, $x[0] = 1$.

**[18:51 - 18:52]** 

**[18:52 - 18:55]** Thus, for $n = 0$,

**[18:55 - 18:57]** The value is 1.

**[18:57 - 19:00]** And this side n is positive.

**[19:00 - 19:01]** 

**[19:01 - 19:03]** P P

**[19:03 - 19:06]** So I can say, x sub j

**[19:06 - 19:08]** So, this is equal to

**[19:08 - 19:13]** $x(0)$ because I know $X(z) =$

**[19:13 - 19:17]** $n = -\infty$ to $\inftyx[n]$.

**[19:17 - 19:20]** $z^{-n}$.

**[19:20 - 19:22]** Now, if you see

**[19:22 - 19:32]** This signal varies from the $n=0$ sample. Thus, it should be replaced with a range that corresponds to the number of samples present in your data set.

**[19:32 - 19:34]** Number 1, 2, 3, 4, 5.

**[19:34 - 19:36]** So, it goes up to 4.

**[19:36 - 19:38]** In the linear form?

**[19:38 - 19:42]** 0 to 5, 0 to 4 is 5 sample.

**[19:42 - 19:49]** So, $x(z) = \sum_{n=0}^{\infty} x[n]z^{-n}$.

**[19:49 - 19:52]** $+ x_1$

**[19:52 - 19:54]** Into $z^{1}$.

**[19:54 - 19:58]** Plus $x[2] \cdot z^{-1}$.

**[19:58 - 20:01]** $\theta = -2$.

**[20:01 - 20:03]** $X(t) + X(\tau)$

**[20:03 - 20:05]** x

**[20:05 - 20:06]** Great.

**[20:06 - 20:09]** Bloff,

**[20:09 - 20:12]** $x[4] \cdot z^{\alpha}$

**[20:12 - 20:14]** How many?

**[20:14 - 20:22]** 0, 1, 2, 3, 4 after that signal does not exist. So, I do not require I do not have to write down that.

**[20:22 - 20:25]** So, I can say x sub 0 is 1.

**[20:25 - 20:27]** $\mathcal{Z}(z^0)$.

**[20:27 - 20:29]** $+ x_1$.

**[20:29 - 20:33]** $- 2 \mathcal{Z}(z^{-1})$.

**[20:33 - 20:35]** $+ x_u$.

**[20:35 - 20:37]** 2, 3.

**[20:37 - 20:39]** $\frac{1}{2}$.

**[20:39 - 20:41]** $+ \mathbf{x}_n^*$

**[20:41 - 20:47]** $x[3] = 0$.

**[20:47 - 20:51]** $+ \mathbf{x}_n^* = -\mathbf{1}$.

**[20:51 - 20:56]** $x[z] = 1.$

**[20:56 - 21:01]** $-2z^{-1}$.

**[21:01 - 21:05]** $+3z^{-2}$

**[21:05 - 21:08]** $\mathbf{a}(t) = \begin{bmatrix} 0 \\ -\mathbf{a}_1(t-1) \end{bmatrix},$

**[21:08 - 21:10]** $z^{-2}$

**[21:10 - 21:12]** So that is the data.

**[21:12 - 21:15]** Let's say I'm another example.

**[21:15 - 21:18]** Here I have a work site.

**[21:18 - 21:24]** Two sides, this side and this side both, this side n is positive

**[21:24 - 21:26]** This sign n is negative.

**[21:26 - 21:27]** 

**[21:27 - 21:31]** So, when I say the sum is over $n$, it varies from

**[21:31 - 21:38]** $x(j) = \sum_{n=0}^{N-1}$

**[21:38 - 21:41]** Number one sample is one.

**[21:41 - 21:42]** Let's see.

**[21:42 - 21:47]** Or let us say the first sample is the zeroth sample, which is 3.

**[21:47 - 21:49]** So, Dirac's sample is 3.

**[21:49 - 21:52]** This side is $-1$, and this is $-2$.

**[21:52 - 21:55]** I can say that $n$ varies from $-2$ to infinity.

**[21:55 - 21:57]** 2, 0, 1 and 2.

**[21:57 - 21:58]** Pootoo.

**[21:58 - 22:00]** $x[n]$

**[22:00 - 22:03]** It's a good report of mine after this.

**[22:03 - 22:13]** $x(-2) = z^{-2}$

**[22:13 - 22:15]** Floss,

**[22:15 - 22:18]** $x[-1]$

**[22:18 - 22:21]** $(x^{-1})^2$

**[22:21 - 22:24]** $+ x(-)$

**[22:24 - 22:26]** My vector of 0.

**[22:26 - 22:28]** $\mathbf{G}(s) = \frac{s}{s^2 + s + 1}$

**[22:28 - 22:30]** Blas, x of

**[22:30 - 22:32]** I can say 1.

**[22:32 - 22:34]** $g = -1.$

**[22:34 - 22:37]** $\mathbf{b} \cdot \mathbf{x}$ and $\mathbf{x}_1$.

**[22:37 - 22:39]** They've got a minor.

**[22:39 - 22:43]** That is simply $x[-2] = 2$.

**[22:43 - 22:45]** $z^2$.

**[22:45 - 22:49]** Plus $x[-1]$ equals 1.

**[22:49 - 22:51]** Get to the part 1.

**[22:51 - 22:53]** Plus $x[0] = 3$.

**[22:53 - 22:55]** $z^0$.

**[22:55 - 22:59]** Plus $x_1 = 1$.

**[22:59 - 23:01]** Get $\frac{1}{z}$.

**[23:01 - 23:05]** Plus $x[2] = -1 - 1$.

**[23:05 - 23:07]** So, finally it is

**[23:07 - 23:12]** $2Z^2 + Z^1$.

**[23:12 - 23:16]** $+c + \frac{z^{-1}}{1}$.

**[23:16 - 23:17]** Mina's get beautiful.

**[23:17 - 23:19]** All right.

**[23:19 - 23:20]** 

**[23:20 - 23:23]** So,

**[23:23 - 23:24]** 

**[23:24 - 23:28]** And what is the reason of convergent in here and here? ROC?

**[23:28 - 23:32]** So, for which $X(z)$ is finite.

**[23:32 - 23:34]** Now if you see,

**[23:34 - 23:38]** In this case, if $z = 0$,

**[23:38 - 23:41]** Then also x sub j is infinite.

**[23:41 - 23:44]** If $z \to \infty$

**[23:44 - 23:46]** Then also

**[23:46 - 23:48]** $x(z) = 0$.

**[23:48 - 23:51]** So, I can say for this case

**[23:51 - 23:53]** Now, the reason of conversion is

**[23:53 - 23:57]** Entire Z plane, so entire Z plane.

**[23:57 - 24:00]** Except $j = 0$.

**[24:00 - 24:02]** And $z = \infty$.

**[24:02 - 24:09]** Now, in this case, if you see, if $z$ equals $\infty$,

**[24:09 - 24:12]** It is 0, but if $z = 0$,

**[24:12 - 24:15]** Then this is infinite.

**[24:15 - 24:22]** So, I can say in this case the reason of convergent is entire Z plane except

**[24:22 - 24:24]** $K = k$.

**[24:24 - 24:25]** 

**[24:25 - 24:27]** 

**[24:27 - 24:32]** So, I have to explain those Z value for which

**[24:32 - 24:34]** $x(z) = \infty$.

**[24:34 - 24:39]** $X(z)$ exists for all values except those on the unit circle $|z|=1$.

**[24:39 - 24:40]** 

**[24:40 - 24:43]** See you, lovely!

**[24:43 - 24:46]** If $x[n] = \delta[n]$,

**[24:46 - 24:48]** $x(z)$?

**[24:48 - 24:50]** Is it equal to 1?

**[24:50 - 24:51]** 

**[24:51 - 24:52]** 

**[24:52 - 24:57]** $\delta[n]$, where $n = 0$ since $\delta[n]$ is defined as being equal to 1 at $n=0$.

**[24:57 - 25:04]** Since $\delta[n] = 1$ when $n = 0$, then $L^2(n) = 0$.

**[25:04 - 25:05]** 

**[25:05 - 25:15]** Thus, I can say that $x(z)$ is simply the sum from negative infinity to positive infinity: $x[n]$.

**[25:15 - 25:20]** $y(t) = \sum_{n=-\infty}^{\infty} x(n)e^{-jnt}$, where $x[n]$ represents a discrete-time signal and $e^{−jwt}$ denotes the exponential term in time domain.

**[25:20 - 25:24]** So, I can say this is nothing but a x sub 0, x sub 0.

**[25:24 - 25:26]** So, we have to write the equation for the

**[25:26 - 25:29]** $X(0) = 1$, $z^0$.

**[25:29 - 25:31]** So it's nothing but the one.

**[25:31 - 25:38]** Therefore, $x[n] = \delta[n]$ implies $X(z) = 1$.

**[25:38 - 25:43]** Similarly, if I say $x[n] = \delta[n - k]$,

**[25:43 - 25:45]** So, same thing I said.

**[25:45 - 25:47]** $x(z) =$

**[25:47 - 25:50]** $n = \infty$

**[25:50 - 25:54]** $-v_n + u_{n+1} = -x_n^z^{-n}$

**[25:54 - 25:57]** $n-k$

**[25:57 - 26:01]** $\delta[n-k]$

**[26:01 - 26:05]** This equals $1$ when $n = k$, otherwise it is $0$.

**[26:05 - 26:06]** 

**[26:06 - 26:08]** 

**[26:08 - 26:17]** I can say that $x[k] \cdot z^{-k}$ for all valid values of $n = k$.

**[26:17 - 26:20]** So, it is 1 into z to the power minus t.

**[26:20 - 26:22]** Did you get to recall?

**[26:22 - 26:23]** Reynolds.

**[26:23 - 26:25]** So x sub j

**[26:25 - 26:27]** Here, it gets to the point

**[26:27 - 26:28]** 

**[26:28 - 26:30]** 

**[26:30 - 26:31]** 

**[26:31 - 26:34]** Similarly, you can do this also.

**[26:34 - 26:36]** So what is eun?

**[26:36 - 26:39]** $u[n] = \begin{cases} 
1 & \text{if } n > 0 \\
0 & \text{otherwise}
\end{cases}$

**[26:39 - 26:46]** $n = 0$, $u[n] = u[0]$ for all non-negative values of $n$.

**[26:46 - 26:47]** Else why did this go?

**[26:47 - 26:48]** 

**[26:48 - 26:51]** Now this becomes a series.

**[26:51 - 26:54]** You can solve the series.

**[26:54 - 26:55]** Are you going to get a…

**[26:55 - 26:56]** 

**[26:56 - 27:00]** $z$-dominance and representation are valid.

**[27:00 - 27:02]** This signal

**[27:02 - 27:06]** So, if I give you a NA signal or NA system,

**[27:06 - 27:09]** You can represent in

**[27:09 - 27:10]** Newton.

**[27:10 - 27:13]** Impulsive response is given.

**[27:13 - 27:17]** So let's instead of signal if I give you a system

**[27:17 - 27:21]** Impulse response of a system.

**[27:21 - 27:23]** Let us say $H(j\omega)$,

**[27:23 - 27:33]** $H(n) = \sum_{n=0}^{\infty} (2 - \delta_n)(-1)^{n+1} + 3(-\delta_0)$

**[27:33 - 27:36]** In minor, 2.

**[27:36 - 27:40]** So, we have to write down the equation for the

**[27:40 - 27:43]** So, this is the equation of the equation.

**[27:43 - 27:45]** This is the impulse response.

**[27:45 - 27:47]** Of 18 of a system.

**[27:47 - 27:51]** Now when the jet does come, .

**[27:51 - 27:57]** $\Delta[n-1] = -2z^{-1}$.

**[27:57 - 28:01]** That $\delta[n-k]$ with $dt$ is what?

**[28:01 - 28:04]** Plus $3z^{-2}$.

**[28:04 - 28:07]** Plus $z$ with exponent one.

**[28:07 - 28:10]** Plus $6 = 0$.

**[28:10 - 28:14]** So, I can say the system get that form is

**[28:14 - 28:18]** $2z^{-1} + 3z^{-2}$

**[28:18 - 28:21]** $\text{plots}(t)^1)$

**[28:21 - 28:22]** 

**[28:22 - 28:24]** 

**[28:24 - 28:31]** So, this is a z plane representation. Now, if I told you what is the region of convergent ROC?

**[28:31 - 28:37]** If $z = 0$, then also it is $\infty$. So, this is $8/z$.

**[28:37 - 28:47]** If $z = 0$, then also $H(z) = \infty$. If $z = \infty$, then also $H(z) = \infty$. So, I can say ROC

**[28:47 - 28:50]** This is the entire z-plane except $z = 0$.

**[28:50 - 28:53]** And $z = \infty$.

**[28:53 - 28:54]** 

**[28:54 - 28:55]** 

**[28:55 - 29:00]** This is referred to as the z-transform.

**[29:00 - 29:10]** There is a property known as the region of convergence (ROC), which excludes any poles from this concept that I will elaborate on later.

**[29:10 - 29:12]** Why not talk about the application of the transfer?

**[29:12 - 29:18]** Now, see that for finite duration of the signal, if the signal is causal,

**[29:18 - 29:27]** I have already said what a causal signal is, and what an anti-causal signal is. This means that if the signal exists only on one side of zero time.

**[29:27 - 29:28]** 

**[29:28 - 29:34]** $n = 0$ on this side and $n > 0$.

**[29:34 - 29:36]** So here m is positive.

**[29:36 - 29:43]** Then I can say $Y(z)$ has a z-transform of always $Z^{-something}$, which means

**[29:43 - 29:48]** At the exception where $z = 0$, the entire z-plane is the ROC (Region of Convergence).

**[29:48 - 29:49]** 

**[29:49 - 29:50]** 

**[29:50 - 29:52]** 

**[29:52 - 29:57]** Similarly, the $NP\text{-}quadl$ signal exists on this side.

**[29:57 - 29:58]** 

**[29:58 - 30:00]** Then I said

**[30:00 - 30:03]** $z^2 + u(t)$

**[30:03 - 30:05]** So I can say it is

**[30:05 - 30:09]** ROC will be the entire z plane except z is infinite.

**[30:09 - 30:12]** $J = \infty$.

**[30:12 - 30:16]** Nyquist's sampling theorem: "The Nyquist rate of a discrete-time signal must be at least twice the highest frequency component."

**[30:16 - 30:18]** In a year.

**[30:18 - 30:21]** Here two sided signal

**[30:21 - 30:28]** The ROC is $z$-plane except $z = 0$ and $z \to \infty$.

**[30:28 - 30:31]** For a finite duration of the signal.

**[30:31 - 30:34]** Might be the signal is infinite, do you listen?

**[30:34 - 30:35]** 

**[30:35 - 30:37]** 

**[30:37 - 30:38]** 

**[30:38 - 30:40]** 

**[30:40 - 30:45]** $X(j)$

**[30:45 - 30:48]** Is a series of infinite sum.

**[30:48 - 30:49]** 

**[30:49 - 30:50]** 

**[30:50 - 30:51]** 

**[30:51 - 30:53]** 

**[30:53 - 30:54]** 

**[30:54 - 30:56]** 

**[30:56 - 30:57]** For if it is

**[30:57 - 31:04]** Thus, $z^{-1}$ and $z^{-2}$ like that.

**[31:04 - 31:05]** 

**[31:05 - 31:07]** 

**[31:07 - 31:08]** 

**[31:08 - 31:18]** In case of an anti-causal system, it would be $z^1 + z^{n-1} + \cdots$. So, $z^1$ plus $z^{n-1}$ plus $\cdots$.

**[31:18 - 31:20]** 2 plus dot z to the power n.

**[31:20 - 31:22]** For this purpose.

**[31:22 - 31:25]** Why little is my name?

**[31:25 - 31:26]** Think about it.

**[31:26 - 31:30]** Now, this purpose $x[jn] =$

**[31:30 - 31:37]** $Z^{-1} + Z^{-2} + \ldots + Z^{-\infty}$ finally, it will be finite.

**[31:37 - 31:38]** 

**[31:38 - 31:39]** 

**[31:39 - 31:41]** 

**[31:41 - 31:42]** 

**[31:42 - 31:43]** 

**[31:43 - 31:45]** I love them.

**[31:45 - 31:47]** So I can say that.

**[31:47 - 31:55]** That I can find out the z-transform of, but if it is a two-sided signal, so instead of an entire impulse response, it will be on a ring.

**[31:55 - 31:56]** 

**[31:56 - 31:58]** 

**[31:58 - 31:59]** 

**[31:59 - 32:01]** We will do one example for that.

**[32:01 - 32:04]** $V_n!$

**[32:04 - 32:05]** 

**[32:05 - 32:09]** For stability, ROC include

**[32:09 - 32:11]** So, unit circle in the genteel.

**[32:11 - 32:13]** Okay.

**[32:13 - 32:15]** For next class,

**[32:15 - 32:20]** I will talk about the properties of z-transform problems, okay. Thank you.

**[32:20 - 32:21]** 

**[32:21 - 32:22]** 

