# Academic Lecture Transcript: Electrical Engineering 100 - Lecture 10 - audio only. No Visu-PGbvb3Pv570(1).wav

**Date:** 2026-10-31 10:43 | **Segment Tested:** 00:00 - 05:00 | **Engine:** Whisper-Small + Qwen2.5-1.5B (CUDA SM87)

---

**[00:00 - 00:01]** Alright!

**[00:01 - 00:10]** As far as I understand, you guys are just starting this, right? This is my slide deck. It's identical. It just has different colors. It's blue.

**[00:10 - 00:13]** And you must have just finished the IC thing.

**[00:13 - 00:17]** All right, all the wonders of wafer silicon fabrication.

**[00:17 - 00:19]** No.

**[00:19 - 00:21]** Wow

**[00:21 - 00:23]** We're going to hide.

**[00:23 - 00:28]** All of this over here

**[00:28 - 00:33]** Yay!

**[00:33 - 00:35]** All right.

**[00:35 - 00:38]** Awesome.

**[00:38 - 00:40]** Are there any questions for me?

**[00:40 - 00:43]** No, I can just start talking shit.

**[00:43 - 00:46]** That's it? Nothing? Yeah, what?

**[00:46 - 00:54]** For real? Don't you read your fucking emails? I even got his email.

**[00:54 - 00:56]** Come on, dude.

**[00:56 - 00:59]** So you knew who I was.

**[00:59 - 01:09]** You just wanted to make a point that I had to introduce myself. So who looks like the shithead now? You or me?

**[01:09 - 01:10]** I'm sorry.

**[01:10 - 01:16]** That's Fawaz, who is a balding old guy, but he's like Papa Smurf, you wanna hug him?

**[01:16 - 01:19]** Doesn't he look like Paul Smirnf in the thing? He's a really cool guy actually.

**[01:19 - 01:24]** He's one of the fathers of EM or whatever. He's a pretty cool guy. Yeah, is he out of your hand up?

**[01:24 - 01:25]** Thank you.

**[01:25 - 01:26]** Can you guys see the answer?

**[01:26 - 01:29]** Can you license them?

**[01:29 - 01:33]** I'm sorry.

**[01:33 - 01:38]** Oh, you were asking why

**[01:38 - 01:43]** why we don't give you, like download and give you the answers to all the end of the chapter problems.

**[01:43 - 01:48]** Okay, so hold on, there's a very easy answer to this.

**[01:48 - 01:52]** Go through two versions. By the way, is this actually recorded?

**[01:52 - 02:04]** I don't record 40, that's why I curse so much. That's fine. Hi, everybody. Hi, grandma. Anyway, look, so...

**[02:04 - 02:09]** The answer is these guys don't, I mean think about it, the textbook, right? So you don't...

**[02:09 - 02:16]** You don't want to give out the answers to these things because then they will go all over the web and everyone will have them except...

**[02:16 - 02:19]** Obviously that's already true unless you

**[02:19 - 02:21]** you don't know what torrenting is so

**[02:21 - 02:23]** Good.

**[02:23 - 02:26]** Does that answer your question?

**[02:26 - 02:29]** All right. Okay, any other questions?

**[02:29 - 02:31]** Thanks.

**[02:31 - 02:32]** What?

**[02:32 - 02:34]** But if you guys find it on the

**[02:34 - 02:41]** No, I think it's I can't I we can't right because the the company would never agree to

**[02:41 - 02:51]** to give out all the answers to the end of the book if some people are assigning them because then they'd go out and wander. I'm not arguing that the position is tenable, it's just that is the position.

**[02:51 - 02:54]** We can't just distribute these things.

**[02:54 - 02:57]** willy-nilly because they are the answers to problems that

**[02:57 - 02:59]** supposedly are guarded by the teacher's manual.

**[02:59 - 03:05]** This is a conflict between a modern reality with information and older models of things right a long time ago

**[03:05 - 03:09]** There was a teacher's manual that there was no way you guys could get a hold of.

**[03:09 - 03:12]** But they're still operating under that model.

**[03:12 - 03:13]** We're going to ask you to move forward.

**[03:13 - 03:15]** Thank you.

**[03:15 - 03:20]** You mean the numbers, the answer is three.

**[03:20 - 03:21]** I'll email NTS.

**[03:21 - 03:25]** I don't set those rules, so I'll email NTS, right? They own the...

**[03:25 - 03:28]** They're the ones that publish it. I'll ask.

**[03:28 - 03:30]** I doubt the answer will be yes.

**[03:30 - 03:35]** But it'll get me out, you'll never see me again so it'll get me out of today, right?

**[03:35 - 03:40]** I promise I'm gonna look into it. I'm gonna look into your problem. Give me a baby kiss. Bye.

**[03:40 - 03:43]** All right, any other questions?

**[03:43 - 03:47]** No, we're good. Should I start? All right.

**[03:47 - 03:48]** So, um...

**[03:48 - 03:54]** I guess you guys are about to do amplifiers in particular op amps and

**[03:54 - 04:01]** Oh, that's true. I don't know what, sorry, that reminds me. I don't know what Vivek does, but in 45 minutes we'll take a stretch break, because otherwise I'm going to fall asleep.

**[04:01 - 04:06]** So just you know, okay, so you guys are about to do amplifiers

**[04:06 - 04:16]** You probably used op amps in the lab if he's following labs similar to ours, and probably frustrated you because he hadn't talked about them before he lectured on them.

**[04:16 - 04:21]** The reason for that by the way is on purpose for 40 which is that

**[04:21 - 04:30]** I in particular believe that you guys should run into this stuff and get you guys I don't know you guys different priorities in the guy seriously than the people taking it as part of their major and

**[04:30 - 04:40]** But it's good to encounter this stuff before I spoon feed you like little baby birds all of this stuff so it forces you to just confront it and fight it a little bit.

**[04:40 - 04:42]** and feel stressed by it, stress is good.

**[04:42 - 04:52]** And then you generate all these questions that were generated by your anger at us and the professors in Dickhead and all this stuff. And then you show up and then we lecture. So there's a reason why that was done.

**[04:52 - 04:53]** in the context of 4D, by the way.

**[04:53 - 04:57]** So anyway, you're doing amplifiers and

**[04:57 - 05:00]** It was a good point here to stop

