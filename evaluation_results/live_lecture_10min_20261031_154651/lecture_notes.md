# Live Test & Lecture Transcription Notes

- **Session ID:** `session_20261031_154651`
- **Started:** 2026-10-31 15:46:51
- **Platform:** NVIDIA Jetson Orin Nano (8GB) | 100% Offline Edge
- **Models:** Whisper small.en (CUDA Flash Attention) + Qwen2.5-1.5B (CUDA SM87)

---

## Transcribed Lecture Content

**[00:08]** In which we will talk about how to calculate the poles of any transfer function $H(s)$.

**[00:15]** So, any transfer function $H(s)$ can be written as $\frac{numerator}{denominator}$. Let's suppose

**[00:23]** The numerator is

**[00:26]** $n(s)$ and the denominator is $d(s)$.

**[00:31]** Now,

**[00:33]** We find the poles of the transfer function $H(s)$ by finding the roots of the denominator.

**[00:41]** The office.

**[00:42]** So, we solve $d(s) = 0$ and whatever the values of $s$ that we get from solving.

**[00:50]** These equations are ultimately the roots of the denominator of $H(s)$.

**[00:00]** This is the topic of time scale.

**[00:08]** And operation that we perform on sequences.

**[00:13]** Okay, and this usually comes in the context when you are changing

**[00:20]** So, we are changing the sampling rate of a discrete-time sequence. But again, we will talk about the change of sampling rate when we talk about

**[00:28]** So, let us capture it in the following form. Let me give you a sequence. This is a sequence

**[00:35]** $P(x[n])$.

**[00:37]** $\theta$, which is many zeros, and then finally starts with $1, 2,$ etc.

**[00:46]** Three, four, five.

**[00:50]** And then four, three, two, one, and then zeros.

**[00:55]** Okay.

**[00:56]** This is a finite length sequence. Is it causal?

**[01:02]** Okay, where is the origin? I didn't show you the origin. Okay, the origin is here.

**[01:08]** So, a finite-length sequence is causal and it starts with the sample value equal to 1.

**[01:15]** Let's sketch it for us here.

**[01:18]** So, we have this particular sequence. It starts at $n = 1$.

**[01:24]** Then $n = 2$, $3$, $4$, and $5$.

**[01:30]** And then it comes down to four.

**[01:34]** Three.

**[01:35]** and then one okay zero one two three four five six

**[01:43]** These are nonzero samples, after that it is zero.

**[01:49]** And the heights we can indicate.

**[01:52]** As

**[01:56]** One, two, three, four, five.

**[02:00]** One, four, three, two, one. Okay, so you are comfortable with this.

**[02:05]** Now, the first operation of time scaling is a reduction in the sampling rate.

**[02:12]** What we are going to say is, I will define the sequence $y[n]$ that is equal to $x[2]$.

**[02:20]** All in.

**[02:22]** Okay, so $y(0)$ will be $x(0)$, $y(1)$ will be equal to $x(1)$.

**[02:30]** Cool.

**[02:31]** $y(2)$ will be equal to $x(4)$, so just write down a few of those. So, $y(0)$.

**[02:37]** One will be $x(0)$.

**[02:39]** $y(1)$ will be $x(2)$.

**[02:44]** And $y(2)$ will be equal to $x(4)$ and so on.

**[02:51]** What does this sequence look like? So basically we will take this.

**[02:56]** At the same sequence.

**[02:59]** And paste it over here.

**[03:04]** With a different color.

**[03:06]** Let's say we'll make it blue and what we have done is we have erased

**[03:13]** It's some of these samples.

**[03:17]** Okay, and we have now renumbered them as

**[03:23]** This is one, two, three, four.

**[03:27]** Okay.

**[03:28]** Or maybe it's easier if you can look at it as a sequence.

**[03:34]** As a sequence.

**[03:40]** And

**[03:40]** Okay, so $y[n]$

**[03:45]** So $y[n]$ is...

**[03:47]** $x[2n]$ and $y(0)$ are the same.

**[03:51]** This sample got dropped. Next sample got dropped.

**[03:55]** And this sample got dropped, this one got dropped and zeros are there. Okay?

**[04:03]** This is obtained by this process.

**[04:10]** This is basically we are doing a scaling of time. This is a process that we call as down-sampling or compression.

**[04:18]** Let's check it.

**[04:19]** So, in general, if you write down an expression, $y[n] =$

**[04:26]** $X[nm]$, where $m$ is an integer.

**[04:33]** Okay, and the value meaning, something greater than 1, 2n or 3m equals 2, 3, 4 and so on. You are basically going to be doing

**[04:41]** Okay.

**[04:42]** A compression, so this is a operation on time scaling which is called as compression or as

**[04:50]** Don't sample.

**[04:51]** I like the term downsampling much better than compression.

**[04:56]** Because, you know, it's very easy to see that you originally had eight non-zero samples, now it went down to five because some non-zero samples got eliminated.

**[05:04]** Some got omitted.

**[05:07]** Okay, now if I were to ask you to write down

**[05:12]** $y_1[n] = x[3n]$

**[05:18]** Okay, this would be a sequence.

**[05:21]** Lots of zeros. You'll start with one.

**[05:25]** Next sample that will be present in your sequence, you are going to throw out two samples and retain the

**[05:33]** Next one. So, two and three got dropped, four will be the next sample. Then five and four got dropped, three will be

**[05:40]** Next are the next non-zero samples, and then two and one, and then zeros everywhere. So this would be the sequence.

**[05:48]** You would get $x(3)$ and a few fewer non-zero samples because you are doing the downsampling.

**[05:56]** Then operation.

**[05:57]** Okay, so anytime we do a down sampling operation there must be a counterpart and yes there is an

**[06:05]** So, we have a sampling operation as well or a expansion operation basically.

**[06:13]** Before I forget, this has got a notation. We say that you take $x[n]$.

**[06:21]** Pass it through a down sampler marked by a down arrow and the factor M.

**[06:28]** And then this output is $y[n]$.

**[06:32]** And

**[06:34]** The output is $y[n]$, which is equal to

**[06:37]** Now moving on to the expansion operation or the

**[06:45]** And sampling.

**[06:47]** Expansion or up sampling.

**[06:52]** Very, very simple. What we are doing here is taking the samples that we

**[07:00]** We already have and are going to insert zeros. Now the question is how many zeros are we to insert?

**[07:08]** So here is an example, as usual. If $y[n]$, I want to upsample by a factor of 2.

**[07:16]** The definition of the function is the same.

**[07:24]** This is equal to $\frac{x[n]}{2}$.

**[07:31]** If, if,

**[07:36]** $n = 0 - 2 - 4 + \ldots$

**[07:43]** Remember, a discrete-time sequence is defined only when the argument is an integer.

**[07:51]** And it is equal to zero otherwise.

**[07:57]** This equals zero otherwise.

**[08:04]** $\theta$ will be the same as $x(0)$.

**[08:07]** $y(1)$.

**[08:08]** You will get $x(1) = \frac{1}{2}$ that is not allowed. Therefore, that sample will be equal to 0.

**[08:14]** $y(2) = x(1)$, yes, that is an integer, you will get equal to $x(1)$.

**[08:19]** $y(3)$.

**[08:21]** Y of four?

**[08:24]** $y = 2$. Now you know the trend, sorry, $x(2)$.

**[08:29]** And you can do the same thing over for all the samples. So the original sequence that we had

**[08:36]** $x[n]$

**[08:38]** At many poles.

**[08:40]** And then started out with a sample one, two, three, four, five.

**[08:47]** Four, three, two, one, and then zeros.

**[08:52]** Okay, so...

**[08:54]** Let $y[n]$ be played down.

**[08:57]** This sample was $y(0)$, which is the same as $x(0)$, but this is the origin.

**[09:05]** The next example of $y[n]$ was a zero.

**[09:10]** Then you had the value equal to $2$.

**[09:13]** Then zero three, zero four, zero five, zero four, zero three, zero two,

**[09:20]** Then you have two, and zero, one, and then all zeros. So basically put a comma in between. So you have inserted one zero between.

**[09:28]** Then every two set of samples.

**[09:32]** Okay?

**[09:33]** Now, the general case of expansion.

**[09:37]** $y[n] =$

**[09:41]** x of

**[09:43]** And by an integer and upper case n.

**[09:49]** Okay, x of...

**[09:51]** And $\frac{1}{n}$.

**[09:54]** If $n$ is a multiple of, if $\frac{n}{n}$ is an integer, this will be true if $n = 0 + \pm$

**[10:02]** Yes, and...

**[10:03]** And notice that for all other values,

**[10:10]** It will be zero.

**[10:13]** So, for $n = 0$, you will get a non-zero value. For $n = 1, 2, 3$, you are going to get zero.

