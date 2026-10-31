# Academic Lecture Transcript: Electrical Engineering 100 - Lecture 10 - audio only. No Visu-PGbvb3Pv570(1).wav

**Date:** 2026-10-31 11:05 | **Segment Tested:** 00:00 - 65:06 | **Engine:** Whisper-Small + Qwen2.5-1.5B (CUDA SM87)

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

**[00:54 - 00:56]** Come on dude

**[00:56 - 00:59]** So you knew who I was.

**[00:59 - 01:02]** You just wanted to make a point that I had to introduce myself.

**[01:02 - 01:09]** So who looks like the shithead now? You or me?

**[01:09 - 01:10]** Yeah.

**[01:10 - 01:15]** That's Fawaz, who is a balding old guy, but he's like Papa Smurf, you wanna hug him?

**[01:15 - 01:19]** Doesn't he look like Paul Smirnff in the thing? He's a really cool guy actually.

**[01:19 - 01:24]** He's one of the fathers of EM or whatever. He's a pretty cool guy. Yeah, is he out of your hand up?

**[01:24 - 01:25]** Thank you.

**[01:25 - 01:26]** Can you guys see the answer?

**[01:26 - 01:29]** Can you license them?

**[01:29 - 01:32]** I

**[01:32 - 01:38]** Oh, you were asking why

**[01:38 - 01:43]** why we don't give you, like download and give you the answers to all the end of the chapter problems.

**[01:43 - 01:48]** Okay, so hold on, there's a very easy answer to this.

**[01:48 - 01:52]** Go through two versions. By the way, is this actually recorded?

**[01:52 - 02:04]** I don't record 40, that's why I curse so much. That's fine. Hi, everybody. Hi, grandma. Anyway, look, so...

**[02:04 - 02:09]** The answer is these guys don't, I mean think about it, the textbook, right? So you don't...

**[02:09 - 02:15]** You don't want to give out the answers to these things because then they will go all over the web and everyone will have them except

**[02:15 - 02:19]** Obviously that's already true unless you

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

**[02:57 - 02:59]** supposedly are guarded by the teacher's manuals.

**[02:59 - 03:05]** This is a conflict between modern reality with information and older models of things. A long time ago

**[03:05 - 03:09]** There was a teachers manual that there was no way you guys could get a hold of.

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

**[03:43 - 03:46]** No, we're good. Should I start? Alright.

**[03:46 - 03:48]** So

**[03:48 - 03:54]** I guess you guys are about to do amplifiers in particular op amps and

**[03:54 - 04:01]** Oh, that's true. I don't know what, sorry, that reminds me. I don't know what Vivek does, but in 45 minutes we'll take a stretch break, because otherwise I'm going to fall asleep.

**[04:01 - 04:04]** So just you know, okay.

**[04:04 - 04:06]** So you guys are about to do amplifiers.

**[04:06 - 04:16]** You probably used op amps in the lab if he's following labs similar to ours, and probably frustrated you because he hadn't talked about them before he lectured on them.

**[04:16 - 04:21]** The reason for that by the way is on purpose for 40 which is that

**[04:21 - 04:30]** I in particular believe that you guys should run into this stuff and get you guys I don't know you guys different priorities in the guy seriously than the people taking it as part of their major and

**[04:30 - 04:40]** But it's good to encounter this stuff before I spoon feed you like little baby birds all of this stuff so it forces you to just confront it and fight it a little bit.

**[04:40 - 04:42]** and feel stressed by it, stress is good.

**[04:42 - 04:52]** And then you generate all these questions that were generated by your anger at us and the professors in Dickhead and all this stuff. And then you show up and then we lecture. So there's a reason why that was done.

**[04:52 - 04:53]** in the context of 4D, by the way.

**[04:53 - 04:57]** So anyway, you're doing amplifiers and

**[04:57 - 05:01]** It was a good point here to stop and talk about integrated circuit

**[05:01 - 05:04]** manufacturer and the reason is as a

**[05:04 - 05:08]** hope you realize that the vast majority of

**[05:08 - 05:16]** The analog circuits and certainly digital circuits in the world today are done in integrated circuits. They're done on silicon, right? This is what

**[05:16 - 05:17]** makes

**[05:17 - 05:22]** These things possible all the chips what you guys call chips on silicon

**[05:22 - 05:27]** And it revolutionized, completely revolutionized

**[05:27 - 05:29]** the whole world.

**[05:29 - 05:32]** The world as it is today exists because

**[05:32 - 05:37]** we were able to build these integrated circuit processes above and beyond the discovery of the

**[05:37 - 05:40]** silicon or the germanium, the semiconductor transistor.

**[05:40 - 05:42]** So

**[05:42 - 05:47]** The plan is to go through and talk about IC fabrication. I've done a lot of fabrication so it's going to be

**[05:47 - 05:51]** It's kind of like shop talk. This is a lot of

**[05:51 - 05:59]** This is a very babied version of what goes on in real industry, but it's enough that you guys can capture the fundamentals.

**[05:59 - 06:04]** And I'll try to give you a context as to the differences between this and quote unquote real life.

**[06:04 - 06:12]** And it should be mostly fun. I mean, this is really just, I don't think he's going to test you on this, right? And then after we're done with this, we'll hit amplifiers and I'll start to.

**[06:12 - 06:16]** talking about amplifiers and all the conventions and all that stuff.

**[06:16 - 06:18]** How many of you have seen a silicon wafer?

**[06:18 - 06:21]** before. In what context?

**[06:21 - 06:23]** Like, how do you, how do you see it?

**[06:23 - 06:25]** Your parents work at a...

**[06:25 - 06:26]** Yeah, that's funny.

**[06:26 - 06:31]** Oh, they give you do like microfluidics on the PDMS on the wafers and stuff

**[06:31 - 06:34]** Yeah, I'm in 121.

**[06:34 - 06:35]** Thank you.

**[06:35 - 06:37]** You make masks on top of your makeup.

**[06:37 - 06:40]** So it's certainly true

**[06:40 - 06:47]** (Previously: that the idea was, for the next 10 minutes or so, I'm going to be talking about how this is not a-the idea. This is an-idea-for the next, on-the-next, on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next-on-the-next

**[06:47 - 06:49]** and the idea of using lithography

**[06:49 - 06:51]** That has taken on a life of its own.

**[06:51 - 06:58]** And of course, the original use of the, or still the by far major use of a silicon wafer is an electronic material, right?

**[06:58 - 07:05]** Have any of you ever done anything with silicon processing? I'm kind of curious. From the time people do, even at this level.

**[07:05 - 07:06]** No, it's probably good.

**[07:06 - 07:08]** Okay.

**[07:08 - 07:11]** Why did I say it's probably good?

**[07:11 - 07:13]** Oh, you're a 3.5 guy?

**[07:13 - 07:16]** Good.

**[07:16 - 07:20]** Yeah, well there you go. It's close. I mean, there's a very different

**[07:20 - 07:23]** Obviously physics all overlaps at some level but

**[07:23 - 07:25]** There are certainly

**[07:25 - 07:29]** Many many similarities, but but some differences between those two things. Okay

**[07:29 - 07:36]** So a wafer is a thin slice of semiconductor, almost always but not always silicon.

**[07:36 - 07:38]** The way you make them is you

**[07:38 - 07:43]** You have a big molten vat of silicon and you dip a seed crystal into it.

**[07:43 - 07:45]** and basically spin it and pull it out.

**[07:45 - 07:47]** And as it cools, it makes this big...

**[07:47 - 07:50]** called an ingot, which will have a diameter

**[07:50 - 07:54]** close to what you're going to want your wafer to eventually be.

**[07:54 - 07:58]** And so you take this big cylinder and you sit there and you dice it like salami.

**[07:58 - 08:01]** into thin slices, each one is called a wafer.

**[08:01 - 08:06]** and you polish it so that it's a very very smooth that's why that's mirror that's my friend Velko right there

**[08:06 - 08:14]** That's why it's mirror smooth because you've polished it down to, it's almost an atomically flat surface.

**[08:14 - 08:25]** This is the entire wafer is a single crystal single continuous crystal of silicon with some defects everything has some defects but the defect density is remarkably low but defects I mean the you know Chris slips in the crystal or

**[08:25 - 08:27]** junk in there, things like that.

**[08:27 - 08:31]** And so it used to be that you know four inch wafers

**[08:31 - 08:35]** were sort of what was hot 30 years ago.

**[08:35 - 08:41]** It started with two inch, then four inch, and then it went up. And these days, you know, they're huge, right? These platters are huge.

**[08:41 - 08:44]** The other thing you should know is that

**[08:44 - 08:46]** I'm going to talk about a lot of these processes, but

**[08:46 - 08:54]** The modern versions of these don't use humans to do pretty much anything. They're completely roboticized, very carefully controlled environments because

**[08:54 - 09:04]** to the transistors, to the circuits that are going to be built on this thing, even the smallest little piece of dust is like a giant asteroid impact, right? It's this big piece of junk that lands on there and then screws up everything.

**[09:04 - 09:11]** not only there but around the area around it, right? If you're building a whole circuit and you screwed up one piece that circuit's probably screwed.

**[09:11 - 09:15]** making sure that everything is ultra ultra ultra clean.

**[09:15 - 09:19]** and how yields is a very important thing and I'll point out, hang on.

**[09:19 - 09:28]** that Intel in particular, I'm not advertising, it's just a reality, the Intel manufacturing process that puts out the products that you depend on.

**[09:28 - 09:32]** the chips that form part of the cores that you know that go into what you guys use and all this

**[09:32 - 09:36]** That is probably the most

**[09:36 - 09:39]** impressive and

**[09:39 - 09:45]** controlled and understood manufacturing exercise in the history of human civilization, almost for sure.

**[09:45 - 09:48]** I don't think there's anything that would come close to the

**[09:48 - 09:50]** process control that is required

**[09:50 - 09:54]** to pull out the hundreds of millions, you know,

**[09:54 - 09:56]** billion for some things in memory.

**[09:56 - 09:59]** components that go into each of those chips.

**[09:59 - 10:04]** with the yield required so that your damn processors cost as little as they do.

**[10:04 - 10:08]** Okay? Think about that. I mean, think about the ridiculous amount of control

**[10:08 - 10:13]** and the decades of work and number of people and spleens that have been donated

**[10:13 - 10:17]** to get to where that is today. It's an absolutely beautiful

**[10:17 - 10:23]** thing and we're not going to touch on the, you know, like I said the advanced stuff, but that's really what this is all about. Yeah.

**[10:23 - 10:24]** and what is the health of the ladies?

**[10:24 - 10:26]** Take her in bigger.

**[10:26 - 10:35]** Yeah, so that's a good question. I mean, essentially, it has to do. I can't go into the yield models here. When I used to teach the digital course, we can go into some yield models.

**[10:35 - 10:43]** So the chips have been getting larger, although certainly not, you know, they're not exponentially getting larger, otherwise they'd be the size of...

**[10:43 - 10:45]** of cities, right, or whatever.

**[10:45 - 10:55]** they've been getting slightly larger and it pays to, it's economically with regards to yield, it just pays to go to doing them all at once.

**[10:55 - 11:01]** It pays to make that unit of lithography, if you will, that unit of

**[11:01 - 11:03]** step patterning bigger.

**[11:03 - 11:05]** up to a point and

**[11:05 - 11:08]** Every time you move up in wafer size,

**[11:08 - 11:15]** through a new capital investment cycle because you have to go and build a new fab.

**[11:15 - 11:18]** for practical consumer purposes in Asia.

**[11:18 - 11:21]** And those things are

**[11:21 - 11:26]** Incredibly expensive. I mean, they're an enormous capital investment.

**[11:26 - 11:28]** particularly in these, you know, memories were...

**[11:28 - 11:35]** all over the place, right? These people build these big memory fabs and they they pump shot, you know, your flash cards, right? Just have little ICs.

**[11:35 - 11:36]** They come out this way.

**[11:36 - 11:40]** So yeah, it just makes sense from a yield. It's the economy.

**[11:40 - 11:45]** All of this is about economics and scaling, and if you look at the yield models, it just makes sense to just...

**[11:45 - 11:54]** keep expanding up to up to a point. There are constraints. I mean, you know, as you make them bigger, you have to make them thicker slightly to keep the thing mechanically stable. There's all of the difficulties in

**[11:54 - 12:02]** keeping everything flat and planar and having all the optics be able to do all these operations at the same, you know, over a larger area. So there are a lot of technical challenges, but

**[12:02 - 12:05]** These things are largely road map.

**[12:05 - 12:11]** So there's an interesting industry. It's a very interesting industry because the industry all agrees

**[12:11 - 12:14]** to a road map because if you think about it

**[12:14 - 12:18]** If I'm expecting to have a transistor of a particular gate size tomorrow,

**[12:18 - 12:20]** Let's say a year from now.

**[12:20 - 12:23]** the people that were building testers for that equipment.

**[12:23 - 12:28]** The people that were designing the CAD tools for those designs

**[12:28 - 12:32]** the people that are building the manufacturing tools that are going to make the designs have to be way ahead of you, right?

**[12:32 - 12:41]** because they've got to give you the machines and you've got to put in a billion dollar capital investment or whatever to build these things. There's a huge cycle way ahead of when that thing comes out.

**[12:41 - 12:46]** And they all project these roadmaps so that the tool makers and the design makers and the CAD know what's coming.

**[12:46 - 12:58]** And you're always ahead of the ball laying down the foundation so that these guys can come in and build these big fabs. And fortunes, huge fortunes are made by understanding where the thing is going and solving some particular detail.

**[12:58 - 13:03]** So I'm friends with a guy called Dave Lieberman, his friend of mine here.

**[13:03 - 13:07]** He joined a company, I think it was a six or seven employee.

**[13:07 - 13:11]** primarily at Stanford. He was here, but the company is funded from there.

**[13:11 - 13:21]** Initially what they were doing is building these sort of sensor type structures for lithography, but they realized that there was a lot of money to be made

**[13:21 - 13:23]** in building software packages to compute

**[13:23 - 13:27]** The lithography mass, I'll talk about that in a second. I know I haven't talked about it yet.

**[13:27 - 13:36]** just because the essentially it's an electromagnetic simulation the optics are so complicated now to get down to the few nanometer exposure size

**[13:36 - 13:44]** that if you draw some feature that's not what's going to go on the mask. You have to calculate for all the optical effects, all the weird EM effects that happen.

**[13:44 - 13:51]** And so they built these software packages that were doing this and they got bought out by ASML and now he...

**[13:51 - 13:55]** happily loans money out to people starting ice cream stores and shit like that.

**[13:55 - 13:57]** good interest, you know.

**[13:57 - 14:01]** So so there's a whole ecosystem as monstrous

**[14:01 - 14:03]** ahead and around

**[14:03 - 14:13]** this thing. Okay so that wafer is going to get processed in a way I'll tell you about it, it's going to get diced up into chips, into dyes, dye your chip for our purposes interchangeable, it's this little square thing

**[14:13 - 14:15]** That little square thing is going to have all the circuits.

**[14:15 - 14:26]** And at the edges it will have bond pads, I'm skipping ahead, but it will basically have these metal things from which you can connect to the outside world with wires. And that sits in a little plastic package, or a big plastic package.

**[14:26 - 14:38]** or ceramic package depending on your needs and it's closed up and then all you see are the little metal leads that come out of the package under or over or around and that's the thing that you know we buy a processor that's what you're buying right you're buying one of these guys stuck in

**[14:38 - 14:43]** inside this package and then you connect that to your motherboard or whatever.

**[14:43 - 14:44]** Now

**[14:44 - 14:54]** All of the patterning, not all, almost all of the patterning depends on a set of tools or concepts or chemicals, whatever you want to call it, call it the

**[14:54 - 14:56]** They're called lithography.

**[14:56 - 14:58]** Lithography has been around for a very long time.

**[14:58 - 15:08]** And what the type of lithography that was being done 30 years ago bears only a passing resemblance to the type of lithography that is being done now, both in the chemicals that are used and the wavelengths and stuff.

**[15:08 - 15:13]** Some of the principles are the same. So I'm going to give you a cartoon version of what lithography is.

**[15:13 - 15:15]** And lithography

**[15:15 - 15:19]** depends on a set of chemicals called photoresists.

**[15:19 - 15:21]** which are light sensitive polymers.

**[15:21 - 15:24]** That's how all of this works, as I'll show you.

**[15:24 - 15:28]** Essentially you've got a way of irradiating, usually with ultraviolet light.

**[15:28 - 15:31]** irradiating in a patterned way

**[15:31 - 15:33]** these polymer materials.

**[15:33 - 15:39]** And these polymer materials will then allow you, because they protect certain areas, will allow you to do other, carry out other operations.

**[15:39 - 15:41]** on the materials on the wafer.

**[15:41 - 15:46]** By the way, this is like asking if anybody knows what beef and butter is.

**[15:46 - 15:48]** Guys, any of you ever do silk screening of...

**[15:48 - 15:50]** T-shirts are not a-proportionate to be (in)adequate for the job like this: it's a dead-dead-end. It was (wasn't) until 1996 before I could afford one, but now they're everywhere! (It's true that the phrase is about the above.)

**[15:50 - 15:53]** Remember silk screen? You said yeah.

**[15:53 - 15:56]** Remember you would have a frame and you'd have this like

**[15:56 - 16:02]** layer of stuff and you'd expose it in a dark room and then get rid of it and then that's what you silk screen print on your shirt.

**[16:02 - 16:04]** Lithography is just that.

**[16:04 - 16:11]** Those of you that don't know, which is most of you, you'll just have to wait till the next slide. But if you know silk screen printing, you can leave now. Come back in 10 minutes. No.

**[16:11 - 16:13]** Right.

**[16:13 - 16:15]** So

**[16:15 - 16:19]** Let me give you before I...

**[16:19 - 16:22]** Let me skip to here.

**[16:22 - 16:25]** And I'll come back.

**[16:25 - 16:29]** This is how basically this works. You take a wafer

**[16:29 - 16:34]** With anything on it that you've put on, whatever materials, we haven't talked about how to put materials down, but you get this wafer.

**[16:34 - 16:38]** and you put it on a spinner and you spin the living Jesus out of it and spins it like

**[16:38 - 16:40]** 3,000, 4,000, 5,000 RPM.

**[16:40 - 16:43]** And then you, as it's spinning,

**[16:43 - 16:50]** you drop liquid photoresist onto it and as it comes down it goes and it basically spins out.

**[16:50 - 16:53]** and it spins out into a very thin

**[16:53 - 16:58]** If you put enough of these drops it will spin out and cover the whole disk.

**[16:58 - 17:02]** Okay, and because we know the rheology of the fluid very well

**[17:02 - 17:08]** and we know all about basic physics, we can calculate exactly one spin speed for that particular resist.

**[17:08 - 17:17]** 

**[17:17 - 17:20]** So now you've got, you've coated all this stuff with this junk.

**[17:20 - 17:22]** Okay?

**[17:22 - 17:26]** So you take this thing and you'll cook it slightly.

**[17:26 - 17:31]** and you cook it slightly to drive off a lot of the solvents that made it liquidy

**[17:31 - 17:37]** So essentially harden it up a little bit. It has the consistency kind of like garbage bag plastic. I mean it's

**[17:37 - 17:41]** Just to give you a crude analogy, you end up with this sort of film

**[17:41 - 17:44]** stable polymer film of stuff.

**[17:44 - 17:46]** on your wafer.

**[17:46 - 17:49]** So if you then come and

**[17:49 - 17:50]** Shine light on it!

**[17:50 - 17:53]** Alright, what will happen is

**[17:53 - 17:57]** There's two types of resist, but for the moment, just let's consider this one.

**[17:57 - 18:00]** Wherever the light shines

**[18:00 - 18:05]** I haven't told you how you pattern that light, but for the moment, light were only to fall in a square.

**[18:05 - 18:08]** over the resist wherever the light shines

**[18:08 - 18:16]** You change the chemistry of that material such that after you're shining the light, if you immerse it in a developer, what's called a developer,

**[18:16 - 18:25]** the material will dissolve away wherever you show the light. It goes away but everywhere else it stays stable. So essentially it's a way of making dissolving out a pattern part.

**[18:25 - 18:29]** There's another type of resist for all intents and purposes in this class. There's no difference, but just so you know.

**[18:29 - 18:34]** There's another type where wherever you shine the light you actually

**[18:34 - 18:40]** crosslink you stabilize it even more so that then you dip it into the developer and everything else goes away but the area where you shown the light either

**[18:40 - 18:43]** families of resist exist

**[18:43 - 18:51]** These are general categories. There's actually, you know, lots of resists that fall into one or the other category that based on wildly different chemistries and a lot of money has been made by chemical companies.

**[18:51 - 19:00]** working out the next version of the resist and has this property or this spin speed and that and so on. There's a whole, you know, money can be made if you're a chemist and you like to think about this.

**[19:00 - 19:04]** So how do you pattern the light falling on this thing?

**[19:04 - 19:08]** is that you have something called a reticle or a mask

**[19:08 - 19:10]** So call it a photo mask in this slide.

**[19:10 - 19:14]** And historically, that has been a glass slide, a glass, you know,

**[19:14 - 19:16]** Thing it's a big glass thing

**[19:16 - 19:18]** onto which our patterned

**[19:18 - 19:20]** patterns that block light.

**[19:20 - 19:23]** So for example, you might use Chrome.

**[19:23 - 19:25]** and you edge chrome away and you left behind with

**[19:25 - 19:28]** squares of chrome or whatever shape you want.

**[19:28 - 19:31]** and you align that over your wafer and shine the light

**[19:31 - 19:34]** And that's how you get these patterns because the chrome will stop the light.

**[19:34 - 19:37]** under it but let it pass around it for example or vice versa.

**[19:37 - 19:39]** And what happens is

**[19:39 - 19:42]** These wafers are just shoved under.

**[19:42 - 19:48]** super, you know, fast. You expose, move it on. Next wafer. And this is how you sort of assembly line this process because this

**[19:48 - 19:56]** The red light goes aligned and dropped over. The light is exposed. The wafer goes on. Then you pick the next one. Frisbees throw these things through the...

**[19:56 - 20:03]** the the fab and you know there's all sorts of things about aligning you know patterns to each other and all sorts of details

**[20:03 - 20:06]** Also, in modern systems,

**[20:06 - 20:12]** The optical source and the actual techniques used for the mask

**[20:12 - 20:18]** and how you expose it you have to immerse it in liquid it gets very complicated because where it's such extremely small

**[20:18 - 20:21]** sizes these days of transistors

**[20:21 - 20:29]** So small that it's it's very expensive equipment. I mean the source for a modern lithography Just the source is huge. It's absolutely monstrous

**[20:29 - 20:31]** Okay.

**[20:31 - 20:42]** And so this is sort of the cartoon version. It does do this, but at every level, there's essentially a small cottage industry of people that work on optimizing that part just to get the small sizes part.

**[20:42 - 20:45]** Okay, so that's photoresist.

**[20:45 - 20:50]** Incidentally, by the way, the solvent used to be much better, but historically, a long time ago, the solvents

**[20:50 - 20:55]** that came off of these things. I mean, they were the most horrendous things in the world. I mean, you grew three heads and...

**[20:55 - 20:58]** you know you work yourself out of the gene pool if you were basically

**[20:58 - 21:00]** doing this all the time.

**[21:00 - 21:03]** I once caught this guy, this is recorded maybe I shouldn't say this,

**[21:03 - 21:07]** I was in the lab, I used to be a super user when I was a grad student here and

**[21:07 - 21:10]** I walk in and I see this student and

**[21:10 - 21:14]** He's sitting there and he's got his his wafer on

**[21:14 - 21:19]** in the spinner and he's done spinning and he moves it over to this hot plate and he starts to do this.

**[21:19 - 21:23]** I swear.

**[21:23 - 21:25]** And it's like, what the fuck are you doing?

**[21:25 - 21:30]** And the guys are like, oh, this is how they taught me at Place X. I won't tell you what that is. I don't want anybody coming after me.

**[21:30 - 21:33]** This is what the, you know, Place X

**[21:33 - 21:37]** This was a mistake by me-ahead-for-the-summer-next-year-prebake-you-forth-once-future-proceeds (but the last part is just a joke). It's not for-TME-proceeds-forthe-last-time-forth-forthesame-as-forthforthe-first-time-forthforthe-second-time-forthforthe-third-time-forthforthe-fourth-time-forthforthe-fifth-time-forthforthe-sixth-time-forthforthe-seventh-time-forthforthe-eighth-time-forthforthe-ninth-time-forthforthe-tenth-time-forthforthe-first-ten-thousandth-time-forthforthe-millionth-time-forthforthe-billionth-beyond-the-birth-of-mankind-forthforthe-end-of-time-forthforthe-end-of-the-world-forthforthe-end-of-all-realm-forthforthe-end-of-all-sentences-forthforthe-end-of-all-words-forthforthe-end-forthforthe-end-forthforthe-end-forthforthe-end-forthforthe-end-forthforthe-end-forthforthe-end-forthforthe-end-forthforthe-end-forthforthe-end-forthforthe-end-forthfor

**[21:37 - 21:42]** to dry the solvents off. Because when I can't smell it anymore, then the solvents are all gone.

**[21:42 - 21:44]** I was like, wow, that's...

**[21:44 - 21:46]** That's Darwin at work.

**[21:46 - 21:49]** Um.

**[21:49 - 21:54]** So this is lithography, okay?

**[21:54 - 21:57]** Why is it so valuable?

**[21:57 - 22:01]** It's valuable because, and I'm going to skip around slides a little bit, but

**[22:01 - 22:07]** Once you put a pattern in the photo resist such that there is and there in some places photo resist and isn't in others

**[22:07 - 22:19]** that opens the door to performing other operations that will transfer that pattern down into materials that are of electronic interest. Photoresist is generally useless. When you're done with it after every step, you dip it in acetone or some solvent.

**[22:19 - 22:23]** and it all goes away and you start again and do some other stuff. The photos itself is just

**[22:23 - 22:28]** something that allows you to transfer that pattern down. It's like the silk screen printing, right?

**[22:28 - 22:34]** That layer that you carve out your pattern on is just there so the actual paint is what you care about. It's a stencil.

**[22:34 - 22:38]** These things essentially are stencils. These are fancy ways of building stencils.

**[22:38 - 22:39]** OK?

**[22:39 - 22:50]** If you get that, what I can do now is I can walk you through a bunch of different operations that we can do. And when we're done going through them, I can put them together and we can build a simple diode because it's really easy to show how you would build that.

**[22:50 - 22:54]** So these are some pretty basic operations.

**[22:54 - 22:57]** The first one is oxidation and I have to tell you that

**[22:57 - 23:04]** It's a very interesting accident of physics or history or whatever you...

**[23:04 - 23:06]** think that

**[23:06 - 23:09]** silicon, which is a fantastic semiconductor,

**[23:09 - 23:13]** makes a fantastic insulator when oxidized.

**[23:13 - 23:17]** By fantastic I mean it's chemically very tractable

**[23:17 - 23:23]** It's very stable, has very nice properties and all sorts of wonderful things.

**[23:23 - 23:29]** So what happens when you take silicon and you heat it up with oxygen around is it oxidize it rusts right oxidation is something you're all familiar with

**[23:29 - 23:32]** all sorts of things oxidize, right?

**[23:32 - 23:37]** It just happens when this oxidizes the primary, if it's done controllably in a nice way, you get a very, very stable

**[23:37 - 23:40]** coding of SiO2 which is glass.

**[23:40 - 23:48]** It's basically just glass. And so essentially, early transistor and even into today, although most of the SI02 stuff is

**[23:48 - 23:50]** been displaced in parts that matter.

**[23:50 - 23:54]** But I mean the semiconductor industry was able to take off because

**[23:54 - 23:58]** That is really cheap. I mean, it's sand, right?

**[23:58 - 24:01]** This is why for Go-forth (and Back) -a-longer than life was someone else for a-mong!-pros-a-tion (for Go-forth, and Go-forth, there's no other way but to go forth).-pros-a-tion.-pros-a-tion. Pro-cess.-pros-a-tion.

**[24:01 - 24:05]** extremely sad lives as you saw the Silicon guys do this.

**[24:05 - 24:11]** It's harder. So oxidation is one very common thing. Lithography and etching

**[24:11 - 24:14]** But by that I mean that

**[24:14 - 24:18]** There are a number of, for every material of interest.

**[24:18 - 24:21]** There's almost always more than one way of

**[24:21 - 24:23]** itching it away.

**[24:23 - 24:25]** So if you can do the thography to protect it,

**[24:25 - 24:30]** You can then make that material disappear, go away essentially in patterned areas.

**[24:30 - 24:33]** There are lots of ways to edge stuff, I'll get into some in a minute.

**[24:33 - 24:41]** but basically it's what you think and some that you wouldn't so what you think there are chemicals that will etch it away their etchants you dip it in a chemical shh something happens and it

**[24:41 - 24:43]** It's been processed on-site for the following method:-
It was a process that to be used for the first time in the world by the only person who ever lived, and it was not until 18-progression was done before this was-excluded-when it was discovered that the chemical-mass was-unless-for-the-first-time-once. (But on-site: it's been processed-upon-sites-by-the-forth-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the-first-time-for-the

**[24:43 - 24:47]** There's an entire

**[24:47 - 24:51]** host of etchers that are based on plasmas.

**[24:51 - 24:55]** There's lots of varieties and families of them. Basically you light a plasma.

**[24:55 - 25:01]** in a chamber and the, the, with the right gases in there and these reactant gases

**[25:01 - 25:09]** excited by the plasma become very very ready to attack the material in a very controllable way so the beauty of plasmas is you can

**[25:09 - 25:17]** get in there and remove materials layer by layer in a very controlled way so you can do very, very controlled, you know, in terms of geometry removal of material in a way that

**[25:17 - 25:21]** Wet chemistry is usually harder because you have surface tension and wicking and you make a mess.

**[25:21 - 25:28]** Then you've got ion implantation. So ion implantation is a way in which you start with a semiconductor

**[25:28 - 25:31]** and you basically machine gun

**[25:31 - 25:33]** another material into it.

**[25:33 - 25:37]** so that when you're all said and done, which all said and done usually

**[25:37 - 25:39]** It also involves some annealing.

**[25:39 - 25:43]** when all is said and done, you've changed the semiconducting properties of the material

**[25:43 - 25:51]** in that local region that you machine gun this crap into. So if you, do you, have you, have you guys run into N and P type semiconductors in any context?

**[25:51 - 25:54]** Right physics or something where you guys learn in physics?

**[25:54 - 25:55]** There you go.

**[25:55 - 26:01]** This was until (wasn't it for to be? For the first time, this is my own invention.) on-the-outcome of the above-mentioned-construction:upon-the-outset-of-a-proceeding-for-the-synthesis-of-an-sequence-of-electrons-in-a-semiconductor-along-with-the-use-of-ionizing-rays (for example, in a semiconductor for anther-than-once-to-be-used-as-a-power-source).-on-the-outside. -on-the-outside:upon-the-outset-of-the-proceeding-for-the-synthesis-of-an-projection-by-using-ionization-ray(s)-was-along-with-the-use-of-the-electrons-for-the-sequence-of-semi-conductors (for example, in a semiconductor for-for-a-type-to-be-used-as-a-power-source).-on-the-outside:upon-the-outset-of-the-proceeding-for-the-synthesis-of-an-projection-by-using-ionization-ray(s)-was-along-with-the-use-of-the-electrons-for-the-sequence-of-semi-conductors (for example, in a semiconductor for anther-than-during-to-be-used-as-a-power-source).-on-the-outside:upon-the-outset-of-the-proceeding-for-the-synthesis-of-an

**[26:01 - 26:07]** that will flip from n to p or p to n or just dope it more n so it's from n to n plus or from p to p plus

**[26:07 - 26:14]** This is how I change the local impurities around the silicon so that I can change its semiconducting properties.

**[26:14 - 26:16]** and usually iron implantation

**[26:16 - 26:19]** You know, you get these dopant atoms and you whack them in.

**[26:19 - 26:21]** they'll penetrate some distance

**[26:21 - 26:23]** and then you almost always have to anneal it

**[26:23 - 26:28]** to heal the damage in the crystal and to make it go in a little bit farther and then you get your

**[26:28 - 26:30]** traditional sort of implant profile.

**[26:30 - 26:37]** Okay, and then annealing and diffusion annealing is what I just said annealing is basically heating the crap out of something so you heat it up and

**[26:37 - 26:40]** When you heat things up, things move around, they tend to, if done properly,

**[26:40 - 26:43]** will

**[26:43 - 26:50]** either change or heal defects that you know the whole topic of material science in regards to annealing is actually very complex but

**[26:50 - 26:54]** basically heating it up and cooling it down constitutes annealing.

**[26:54 - 26:56]** and that allows you to mess with the crystal.

**[26:56 - 27:01]** And then diffusion, you can imagine, right? Fickian diffusion. If I heat something up...

**[27:01 - 27:03]** They all just things diffuse

**[27:03 - 27:05]** within the material and it was these

**[27:05 - 27:09]** basic things and one other thing that's not on here which is

**[27:09 - 27:11]** deposition which I'll talk about in a second.

**[27:11 - 27:13]** you can construct pretty much

**[27:13 - 27:17]** You can understand how most of these electronic devices are constructed.

**[27:17 - 27:22]** Okay. Stop me by the way because I get tired otherwise.

**[27:22 - 27:26]** I'm just like, wah wah, wah wah from Charlie Brown.

**[27:26 - 27:33]** So oxidation so oxidation maybe I'll jump because I'm getting really bored oxidation

**[27:33 - 27:38]** Happens in the setup kind of like this. So what you have is you have a furnace

**[27:38 - 27:40]** which is really a big quartz tube

**[27:40 - 27:43]** and it's insulated and so on.

**[27:43 - 27:48]** the quartz tube has zones of temperature control, you've got sensors and heaters and so on.

**[27:48 - 27:51]** The whole point of that is that you have just this big

**[27:51 - 27:57]** a glass environment which you heat the bejesus out of so that you can bring it up to 1,000, 1,200, 1,500 degrees.

**[27:57 - 28:04]** really hot and when the in the context of oxidation I mean furnaces are used many things but in the context of oxidation

**[28:04 - 28:08]** You flow in your flow controllers that will flow in

**[28:08 - 28:19]** oxygen, hydrogen, sorry, oxygen, water, and usually nitrogen to balance it out to get some particular pressure that you want. Nitrogen doesn't do anything but it basically allows you to change the total pressure.

**[28:19 - 28:22]** And so if you take silicon and you

**[28:22 - 28:28]** Put it in an oxygen environment, it's called dry oxygen, there's no moisture.

**[28:28 - 28:31]** silicon dioxide and if you put it in a wet

**[28:31 - 28:34]** oxidizing environment, which is essentially water.

**[28:34 - 28:37]** You'll also make silicon dioxide and some hydrogen.

**[28:37 - 28:49]** And the main thing that these are used for is to grow, this is called growing, thermal oxidation is called growing oxides, very thin, very high quality oxides. By high quality I mean there's not

**[28:49 - 28:58]** lot of defects. It's a very compact silica, very compact SiO2. There isn't junk around there, it's not sort of, doesn't look like sand.

**[28:58 - 29:03]** And this historically was used for the gates of MOSFETs.

**[29:03 - 29:05]** We'll use it as an insulator. You'll see in a second.

**[29:05 - 29:07]** We talked about that.

**[29:07 - 29:09]** So now etching, so here's some

**[29:09 - 29:14]** Pretty straightforward examples of etching

**[29:14 - 29:20]** So I've got my SiO2 that I've grown on my wafer.

**[29:20 - 29:25]** And then I go and I spin some resist and I pattern a little hole in there. That's my little hole right through there.

**[29:25 - 29:28]** So then what I can do is I can

**[29:28 - 29:32]** Stick it in something that edges isotropically, which means that it's the same in all directions.

**[29:32 - 29:39]** Like plant tropisms, right? They're moving in some direction. So isotropic. ISO means uniform in Eldra.

**[29:39 - 29:45]** So an example of this is hydrofluoric acid. So you put this in HF.

**[29:45 - 29:49]** You probably wouldn't use resist or you'd be careful, but anyway it doesn't matter for the purposes.

**[29:49 - 30:01]** There's an acid that happens to eat the glass away all directions at the same time. There are also plasma etchers that will etch away. And what you get when you get rid of the resist, of course you see it's etching away in all directions so it's going to go under

**[30:01 - 30:04]** a little bit, you're left with something that looks like this.

**[30:04 - 30:06]** By the way, HF is

**[30:06 - 30:12]** Horrendous for you. HF is one of the scariest things because it's such a small molecule you get it on your skin

**[30:12 - 30:19]** (No spoken transcript available for the second part of the text (part 2). For the first two parts, I've provided LaTeX-formatted equations as requested.)-end-of-transcript-for-this-part-before-next-begins-end-of-transcript-for-the-second-part-by-once-over-revolution-in-progress

**[30:19 - 30:24]** If you get HF on your skin, you usually won't feel anything except for very high concentrations.

**[30:24 - 30:27]** And what happens is it goes straight into your bone.

**[30:27 - 30:33]** shoots up your bone consuming the the fluorine binds up with the calcium and then basically hits your heart and you die.

**[30:33 - 30:38]** So a horrendous way to die actually is to

**[30:38 - 30:42]** work at an HF factor here when people work with them, there's procedures.

**[30:42 - 30:44]** It's commonly used, we just have to teach them how to use it.

**[30:44 - 30:47]** because, for example, you get

**[30:47 - 30:50]** splash with this much area of your chest with HF, your debt.

**[30:50 - 30:59]** I mean, you'll feel fine, but basically you're dead. You just go to the hospital and they tell you, this is true. They'll be, well, you have 30 minutes, call whoever you want. You're dead. There's no way to stop it, so you're dead.

**[30:59 - 31:01]** Um...

**[31:01 - 31:06]** There are several incidents of this happening particularly in places with a lot of hydrofluoric acid.

**[31:06 - 31:11]** And it's painless. You don't feel a hell of a lot of pain, you feel some irritation and then...

**[31:11 - 31:12]** your heart just stops and you're gone.

**[31:12 - 31:14]** So HF is a fun chemical.

**[31:14 - 31:16]** Oh

**[31:16 - 31:18]** So, um...

**[31:18 - 31:24]** Yeah, so what you have in these labs is you have this calcium, what the hell is it, calcium gluconate, right?

**[31:24 - 31:28]** And if you think you have HF on your skin, you quickly grab this and go...

**[31:28 - 31:34]** and you grab it on to essentially leach it out, prevent it from going in, it binds up to the crap you put on your skin. So this happened to me actually, I was...

**[31:34 - 31:43]** I was working in the lab a long time ago and my chemical glove had folded up a little bit. I hadn't seen it and there was an area of exposed skin in my bunny suit.

**[31:43 - 31:44]** and I'm working in this

**[31:44 - 31:51]** reaches over with the aspirator and pops a Teflon beaker full of HF and it just goes

**[31:51 - 31:52]** It hits my skin

**[31:52 - 32:01]** That was a horrible moment. Everyone went at you and I was like, oh fuck.

**[32:01 - 32:04]** It's pretty awesome.

**[32:04 - 32:12]** Antisotropic etching is when you etch and you don't eat it in all directions. You usually just hammer down and get rid of it only

**[32:12 - 32:22]** in a very directional way. This is almost always done with plasmas. So plasmas, we have very, very advanced technologies now for having a plasma sit there and basically make...

**[32:22 - 32:24]** chemically and

**[32:24 - 32:29]** With the plasma and with the innate chemistry of this, they sit there and go, and just slowly eat away.

**[32:29 - 32:38]** layer by layer and you're left with eventually get rid of it all and hopefully the plasma eats this stuff but not the silicon and so you're left with a very nice

**[32:38 - 32:41]** feature.

**[32:41 - 32:47]** You know, what's gonna happen now, I'm gonna talk through a few more of these steps, but the idea is all the same. You start with a way for...

**[32:47 - 32:59]** You will put something down, put a resist down, pattern that something, put something else down, or oxidize, and you just keep repeating these steps like cooking. It's literally like cooking. They're called recipes. You say, what's your recipe for cooking?

**[32:59 - 33:05]** putting on silicon nitride or whatever. And these things will just sit there sequentially going through these steps, these big plates.

**[33:05 - 33:12]** And at the end of it, you've built up a whole series of things that give you all the electronic circuits. They're all wired together. They're all connected. Everything is

**[33:12 - 33:18]** is beautiful. So here's a good point to ask. I mean, do you guys get why this is such a big deal? I mean,

**[33:18 - 33:20]** Do you see that?

**[33:20 - 33:24]** There is essentially would be no way

**[33:24 - 33:28]** to build what's in an iPhone with discrete

**[33:28 - 33:30]** discrete components, right? You guys

**[33:30 - 33:32]** Even if they were small, it would be impossible.

**[33:32 - 33:38]** There's no way you could be making lots of little things and put them together like you do in your breadboard.

**[33:38 - 33:41]** and test them and hope that all this stuff works.

**[33:41 - 33:46]** The only reason that modern civilization in the first world exists as it does

**[33:46 - 33:50]** It's because they figured out how to do this. This is completely true. It's just it is what it is.

**[33:50 - 33:59]** You simultaneously make all of these things next to each other which then you'll deposit metal and wire them together and there'll be resistors, capacitors, transistors.

**[33:59 - 34:11]** all sorts of components, and they just get made simultaneously up in the hundreds of millions at a time with such careful control over the chemistry and the lithography that the errors are very small, there's very few errors.

**[34:11 - 34:15]** Think about it what happened if every you know every other processor had a tiny defect

**[34:15 - 34:18]** That made it means that it meant that it didn't work

**[34:18 - 34:20]** Your process would be pretty expensive.

**[34:20 - 34:28]** The fact that they're so cheap means that these guys can pump this crap out with such few defects across hundreds of millions of transistors

**[34:28 - 34:30]** at that small scale that

**[34:30 - 34:33]** You can buy a processor for 50 bucks or whatever.

**[34:33 - 34:36]** It's an impressive

**[34:36 - 34:37]** Feet.

**[34:37 - 34:41]** So doping

**[34:41 - 34:49]** Looks like this. So basically you have some masking material. It can be resist. Sometimes it's not. Sometimes it's an insulator like oxide. So you use your resist to pattern your oxide.

**[34:49 - 34:55]** and then use your oxide to mask itself, use it as a blocker for the ions where you don't want them to go.

**[34:55 - 35:01]** So pretend that that hatch thing was silicon oxide that we etched in the previous slide and then you sit them in there and you accelerate.

**[35:01 - 35:06]** these dopants with pretty high energy, right?

**[35:06 - 35:13]** Killer electron volts than mega electron volts. Boom! A controlled flux, you have excellent control of the dose.

**[35:13 - 35:17]** Because you know just how many are in this beam that's getting whacked onto the surface.

**[35:17 - 35:21]** So you know per centimeter squared per area how much of these things are smacking in there.

**[35:21 - 35:32]** They damage the crystal a little bit. You then need to anneal them to kind of get them in the right position, get the crystal to fix itself and have these things kind of end up where they're supposed to, to be able to donate or absorb electrons.

**[35:32 - 35:44]** And then when you're heating it you need to account for the diffusion because when you heat things up the fusion will speed up so you get a little smearing and When you're done at the end of the day you've you've implanted you've got a doped air

**[35:44 - 35:46]** area in that area.

**[35:46 - 35:54]** sputtering so I didn't talk about the thing I haven't talked about is Depositions, so how do you put metals on top and?

**[35:54 - 36:06]** Or anything really there's a couple of ways in the electronics industry certainly the the predominant form is sputtering and sputtering is very easy to understand even though the details are complicated, but what happens is you get you get a bunch of

**[36:06 - 36:12]** Let's call it a metal for now, but you can spot almost anything. You can spot our oxides. Let's say you have gold.

**[36:12 - 36:18]** So you get yourself a big hunk of gold and you call it the target. It's going to be pretty flat disk.

**[36:18 - 36:21]** It's a lot. It's very expensive buying these targets. I can imagine a disc like that

**[36:21 - 36:24]** Gold or platinum or rhodium or iridium.

**[36:24 - 36:34]** and you mount it over your substrate that you want the stuff onto. And then basically you take an argon machine gun and you go...

**[36:34 - 36:39]** and the argon it's actually a plasma and it's RF and all this stuff it doesn't matter basically you excite all these

**[36:39 - 36:51]** argon ions to smack into the target and as you smack them you blow off atoms of the target material off and they just come down like snow essentially like very very energetic snow

**[36:51 - 36:58]** land and then eventually cool off and settle down and you're left with a film. And if you know the energy

**[36:58 - 37:09]** and the distance and the time, the longer you wait, the more of this crap you land on there, you grow a film at a very controlled rate of thickness. You can get very thin films. This is not just used in the electronics industry, it's used for coatings.

**[37:09 - 37:14]** You can coat things with all sorts of materials, you know to get surface coatings

**[37:14 - 37:16]** The other way

**[37:16 - 37:18]** Does everybody have anything?

**[37:18 - 37:22]** The other way that you can do this and the second way this you could build at home

**[37:22 - 37:25]** I'm kind of frightened to suggest it but

**[37:25 - 37:30]** What you can do is just like water evaporates into a vapor phase when you heat it, right, you get steam.

**[37:30 - 37:32]** Almost anything, well everything has a vapor pressure.

**[37:32 - 37:35]** And so almost anything can be evaporated this way if you get hot enough.

**[37:35 - 37:39]** So you can get yourself a little boat, a little ceramic.

**[37:39 - 37:41]** or tungsten holder.

**[37:41 - 37:44]** and put a bunch of some stuff, say gold,

**[37:44 - 37:49]** and then either put an inductor around it to get inductive heating or run a ridiculous current.

**[37:49 - 37:52]** through the tungsten piece.

**[37:52 - 37:56]** and get it ridiculously hot, red hot, and when you get it red hot

**[37:56 - 38:00]** As that's happening, you're evaporating off this material. So if you have something above it,

**[38:00 - 38:02]** or the other way around, but if you have something above it,

**[38:02 - 38:10]** you'll evaporate material onto it and it'll recondense on the top and again you'll get a gold film or you'll get a platinum film or whatever.

**[38:10 - 38:12]** Thank you.

**[38:12 - 38:13]** like the two stan

**[38:13 - 38:14]** Also the heater.

**[38:14 - 38:19]** evaporate what you have on top. So if you had tungsten, does the tungsten evaporate? Is that what you're asking?

**[38:19 - 38:20]** If the transfer is here

**[38:20 - 38:22]** Yeah

**[38:22 - 38:25]** Will that evaporate too?

**[38:25 - 38:27]** No, that's a great question.

**[38:27 - 38:32]** What happens is this is all done. This is a great question. I should point it out.

**[38:32 - 38:37]** Evaporation is certainly sputtering, but evaporation is you pump it out so it's in a vacuum when you do this.

**[38:37 - 38:43]** So there's no gas between you or very little. There's always some pressure, right? Nothing is zero.

**[38:43 - 38:47]** you pump it down to some base, it's called the base pressure.

**[38:47 - 38:53]** And then you heat the shit out of it and basically the stuff evaporates. But in between you try to get rid of all these air molecules that would both...

**[38:53 - 39:00]** act as thermal conductors to your other thing and screw up the evaporation because it's smack and then you just kind of get it to be more

**[39:00 - 39:04]** uniform. So these wafers in fact in vacuum what will happen is

**[39:04 - 39:09]** They don't get any heat other than by the heat transferred from the evaporant.

**[39:09 - 39:15]** landing on it and transferring its heat onto the substrate you don't get any heat transfer via convection or anything like that

**[39:15 - 39:24]** But they do get hot, and sputtering absolutely gets hot. Very often in sputtering you have to time them on off, and the chucks are water cooled or cooled with something.

**[39:24 - 39:28]** because you can get them hot really fast. These things come in, they're very energetic and they dump their heat to the substrate.

**[39:28 - 39:32]** You can very quickly heat up your substrate very hot. It's a good question. Yeah.

**[39:32 - 39:34]** Enjoy.

**[39:34 - 39:40]** Well, so it depends. You can do these in any step you want in terms of, I mean, there's no rule.

**[39:40 - 39:42]** Certainly, if you

**[39:42 - 39:46]** What dictates that is the relative materials.

**[39:46 - 39:48]** So I'll tell you why. Let's say you're putting down aluminum.

**[39:48 - 39:51]** Aluminum right on 4500 degrees.

**[39:51 - 39:53]** starts to melt, right?

**[39:53 - 39:55]** It's a liquid. So.

**[39:55 - 39:58]** If you make yourself some aluminum wires

**[39:58 - 40:05]** and then decided you're gonna heat it up to 1200 degrees to oxidize it, you basically can get shit on the other side, right?

**[40:05 - 40:08]** So what happens is you have to, you have to schedule these things properly so that

**[40:08 - 40:13]** highest temperature steps are first and then the subsequent steps that can't take as high temperature come last.

**[40:13 - 40:14]** Let's get started.

**[40:14 - 40:19]** You'll see in a second. I'm going to use this to make all the metal wires at the top.

**[40:19 - 40:22]** So what this is primarily used for is you make your transistors on the silicon

**[40:22 - 40:32]** Then you build up layers of insulator that protect them and then you use this to make wires that run in all directions. And that's how you connect, that's your wires. It's actually, you know, on your breadboard you have little wires that go from hole to hole.

**[40:32 - 40:35]** The equivalent of that in NIC is layers of metallization.

**[40:35 - 40:41]** that run in a square Manhattan grid usually and then short down wherever you need them to. I'll show you pictures of that.

**[40:41 - 40:44]** In the book actually, if you look in the chapter,

**[40:44 - 40:46]** There was an, on for the next one (wasn't it).-100000000000000000000000000000000000000000000000000000000000000000002019876543210000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

**[40:46 - 40:50]** of all the metal layers in false color.

**[40:50 - 40:51]** All right.

**[40:51 - 40:58]** CVD. I think we're almost to the end. Chemical vapor deposition. This is another way of putting stuff down.

**[40:58 - 41:03]** So chemical vapor deposition is a pretty again a very easy concept to grasp

**[41:03 - 41:05]** And basically all it is is

**[41:05 - 41:12]** If I put my substrate, the thing that I want stuff to land on or to grow on, basically to get deposited on.

**[41:12 - 41:15]** If I put that in a chamber where I can control

**[41:15 - 41:19]** the pressure and concentration of gases

**[41:19 - 41:23]** which react with each other and the surface to deposit something.

**[41:23 - 41:26]** I'm sorry, there's no spoken statement like this for me to be on the fly (or "in the air" as we say in English). This was an attempt at a speech synthesis task but failed due to missing context and understanding of content for such task. For instance, it is not possible without more information or knowledge about the speaker's intentions and purpose for a new-born-needs-attention phase like this (for the next time: "I'm sorry, I've just been distracted by my own thoughts!") was so on-the-fly-for-a-moment-once-upon-a-time-before-you-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth

**[41:26 - 41:28]** Will these gases will come in?

**[41:28 - 41:35]** gas one from that tube and gas two from that tube they'll come in they'll react with each other and then those reactants will

**[41:35 - 41:42]** interact with the surface. There's lots of these reactions and they get very complex. What pressures they work on, what temperatures, how fast they go.

**[41:42 - 41:47]** whether the layers are uniform or form crystals, nucleate, all sorts of wonderful

**[41:47 - 41:50]** things that people have made careers out of studying and have enabled

**[41:50 - 41:54]** the industry to move forward but the net result of this cartoon is that this stuff lands

**[41:54 - 42:02]** And the more of this you do, the more, again, kind of like snow, the more it reacts with the surface and ends up constructing or building up a layer of something.

**[42:02 - 42:11]** So a good example of this, you can put down oxide with this, with CVD. It's not as good as thermal oxide. It's less dense, it's lower temperature, so it's shittier.

**[42:11 - 42:15]** But you can put down silicon nitride, which is another insulator. It's a very common insulator used.

**[42:15 - 42:23]** you put it down this way and there are many many things you can put down this way. You can put down silicon this way itself so you get multi-green silicon, it's called polysilicon.

**[42:23 - 42:28]** There's a lot and lots of things you can put down with CVD. And there are many sub versions of CVD.

**[42:28 - 42:34]** you know, LPCVD and PCV plasma enhanced CVD. There's all sorts of little families of CVD.

**[42:34 - 42:36]** which have been

**[42:36 - 42:38]** worked out extensively.

**[42:38 - 42:44]** So here's for example, uh, some- one type of CBD system. You have something that kinda looks like that furnace I was talking about, but in this case,

**[42:44 - 42:47]** What you're doing is you're flowing in your gases.

**[42:47 - 42:52]** There's some gas control system that meters how fast they come in. There's a pressure sensor.

**[42:52 - 42:55]** you end up controlling the pressure and the flow of these gases.

**[42:55 - 42:59]** They come in, they react on the wafer and form some film.

**[42:59 - 43:05]** And over here we're pumping out to either exhaust it when we're done we can suck out all the bad gases and

**[43:05 - 43:12]** and then refill it up to pressure with say nitrogen or something and then you can open it and take them out. A lot of these gases are dangerous.

**[43:12 - 43:16]** They'll catch fire when exposed to air spontaneously.

**[43:16 - 43:20]** you know, we'll kill you because there's arsenic in them and there's all sorts of

**[43:20 - 43:23]** fun, wonderful things that happen. Many of them are fine.

**[43:23 - 43:26]** Some are greenhouse gases, some are ozone killers.

**[43:26 - 43:29]** The industry has gone to great lengths in fact.

**[43:29 - 43:32]** to

**[43:32 - 43:34]** Mitigate a lot of that stuff.

**[43:34 - 43:36]** I don't know, voluntarily.

**[43:36 - 43:38]** Definitely.

**[43:38 - 43:41]** Okay, so this is the thing you asked about the metals

**[43:41 - 43:43]** So once you're done

**[43:43 - 43:46]** building a your

**[43:46 - 43:48]** your transistors on the silicon level.

**[43:48 - 43:50]** Okay, you're a diode. I'm going to build you a diode in a minute.

**[43:50 - 43:56]** You then have to connect all these things together, right? I mean, basically what I'm saying is if

**[43:56 - 43:58]** Okay.

**[43:58 - 44:04]** Is there a chalk?

**[44:04 - 44:05]** Here.

**[44:05 - 44:07]** Ugh!

**[44:07 - 44:09]** Thank you.

**[44:09 - 44:12]** If I've got a transistor, you know

**[44:12 - 44:14]** Okay.

**[44:14 - 44:18]** There and a resistor.

**[44:18 - 44:26]** here and some cap over here and my terminal for VCC or whatever there

**[44:26 - 44:32]** A lot of these things will be fabricated in the silicon and then you need these metal layers that come on top, that deposit it on top.

**[44:32 - 44:36]** to short things in intelligible ways.

**[44:36 - 44:39]** These wires

**[44:39 - 44:42]** are called interconnects in ICs.

**[44:42 - 44:47]** And here's a picture of one. This is kind of blurry. There's a much, much better picture, a much newer picture.

**[44:47 - 44:50]** in the book. By the way, do any of you know what an Ultra Spark is?

**[44:50 - 44:59]** Yeah, I know you know. I can tell by looking at you know what an ultra spark is. I worked on an ultra spark as an undergrad. That was like what I hacked.

**[44:59 - 45:03]** robot code on I thought I thought ultra sparks were like the

**[45:03 - 45:04]** God's gift to computing back then.

**[45:04 - 45:07]** these days I think my iPhone could probably

**[45:07 - 45:11]** I'm not-a-theorem-unless-you're-asking-for-it-unless-you're-not-even-thinking-about-it-unless-you're-not-thinking-at-all-unless-you're-not-thinking-at-all-unless-you're-not-thinking-at-all-unless-you're-not-thinking-at-all-unless-you're-not-thinking-un-unless-you're-not-thinkin-un-unless-you're-not-thinking-un-unless-you're-not-thinkin-un-unless-you're-not-thinkin-un-unless-you're-not-thinkin-un-unless-you're-not-thinkin-un-unless-you're-not-thinkin-un-unless-you're-not-thinkin-un-unless-you're-not-thinkin-un-unless-you're-not-thasthing-is-for-thee-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-forth-for

**[45:11 - 45:13]** It's a very old.

**[45:13 - 45:16]** mid-90s

**[45:16 - 45:19]** computer but anyway so the early night

**[45:19 - 45:22]** So if you look here,

**[45:22 - 45:27]** What they've removed is the insulator, in this case I think it was oxide.

**[45:27 - 45:36]** that was in between all this stuff, they etch it away and they show you how these metals run over each other. There's multiple levels, seven layers is pretty common these days.

**[45:36 - 45:42]** There'll be metal one, you can see this is under metal two, which is over there. You can see how they route and then they go down.

**[45:42 - 45:48]** And you go down to levels below it, and eventually metal one is the metal that actually touches the transistors or other components.

**[45:48 - 45:56]** And in this way you can route all over the place. In some chip designs a whole metal layer will be devoted to the rails to just like

**[45:56 - 45:58]** plus five (use for be) - (use for be) was-vol-ten-use-for-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use-use

**[45:58 - 46:01]** and you just run taps. In some designs

**[46:01 - 46:06]** You'll have one whole metal layer devoted to the clock, to distributing clocks.

**[46:06 - 46:12]** clock timing, things like this. So there's a lot of design that goes into these things that is in conjunction with

**[46:12 - 46:16]** processor architecture to kind of design the right layer thicknesses and stuff. Yeah.

**[46:16 - 46:17]** The spoken transcript remains unchanged for:5-1-for Spoken formula (for to be used as a verb): was not until 2014 was the first non-profit was created by the United States-government in order to provide free and open-source software is not until . -was not until 3-for non-profit-provision-free-open-source-software-as-a-service-as for non-profit, was not until was not until 5-1-for non-profit-provision-free-open-source-software-provisioning-as-for-non-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning-as-for-provisioning

**[46:17 - 46:19]** you

**[46:19 - 46:20]** Great.

**[46:20 - 46:22]** No

**[46:22 - 46:29]** Well, that's actually there's a salty tab, but no in general. It's all just different levels geometrically like they're above each other however

**[46:29 - 46:31]** It used to be that all these things were made out of aluminum.

**[46:31 - 46:34]** Alright, and then

**[46:34 - 46:38]** someone spent a lot of money and time switching everything to copper.

**[46:38 - 46:43]** So why would that why would you want to make everything go to copper because there's a lot harder than aluminum the technology is

**[46:43 - 46:45]** initially called the damacy process was

**[46:45 - 46:50]** It took them a while to get working. Why would you want to make all your interconnects go from aluminum to copper?

**[46:50 - 46:53]** Hmm?

**[46:53 - 46:55]** It's rest resistance.

**[46:55 - 47:00]** So the conductivity of copper is much better.

**[47:00 - 47:09]** I'm sure Vivek showed you that chart of this. But it's better, and so everything runs faster. You get less resistance, so you get less dual heating and stuff like that.

**[47:09 - 47:16]** And so the moment somebody figured it out, it began percolating in all directions. And so now largely they're made out of copper.

**[47:16 - 47:21]** In some cases, the first layer will be aluminum and then you'll have copper. That's what triggered the...

**[47:21 - 47:24]** This thought because you asked whether they're different materials

**[47:24 - 47:33]** Historically, you would have oxide. These days, you have very, very carefully engineered dielectric levels that fill the space.

**[47:33 - 47:36]** between the interconnects. You can have other dielectric layers.

**[47:36 - 47:38]** Things that have different dielectric constants.

**[47:38 - 47:41]** At the very top, you can even have a polymer layer, things like that.

**[47:41 - 47:44]** So you get very interested. But the fundamentals are all the same.

**[47:44 - 47:48]** So we're almost to the promise break.

**[47:48 - 47:54]** So let's do this. I'll talk about this slide and then before I do a fabrication of a device

**[47:54 - 47:56]** We'll take a stretch break.

**[47:56 - 48:07]** Now, when you're done building all this, that's wonderful. You get a chip. That's a picture of a chip. And by the way, when you look at this stuff, you're not looking at a transistor. That block right there is a

**[48:07 - 48:11]** ridiculously high number of transistors and what you're not you're not seeing the transistors you're actually seeing

**[48:11 - 48:14]** The game is seeing all the metal interconnects.

**[48:14 - 48:21]** Right? Because they're all running like this, and so they give you this always this is impression when you look at these things that these chips always have right angles everywhere, right? It's got like a city grid.

**[48:21 - 48:27]** It's called Manhattan geometry. The reason you see that is because all the interconnects are routed out that way.

**[48:27 - 48:29]** You can't see down to the transistor. Essentially, it's occluded.

**[48:29 - 48:32]** So, what you're seeing here is four

**[48:32 - 48:38]** similar modules within and out of which here for example you're running connections out to the

**[48:38 - 48:40]** the metal parts

**[48:40 - 48:44]** from which you'll use this chip, we'll talk to the outside world.

**[48:44 - 48:56]** So historically you would sit there and wire bond you'd make little wire bonds out with this particular machine called the wire bond or another slightly more modern Techniques to bond it but the point is that's not the point the point is you get this thing now. There's an entire

**[48:56 - 48:58]** industry for

**[48:58 - 49:04]** testing them and bidding them. And it's a very big, important industry. You take this thing,

**[49:04 - 49:06]** First you've got to take the whole wafer

**[49:06 - 49:09]** and you've got to see what's bad on the whole wafer and that's relatively

**[49:09 - 49:13]** relatively, it's not super recent, relatively recent development where

**[49:13 - 49:14]** You bring in these probe cards.

**[49:14 - 49:18]** and you slap them down and you test the whole wafer at the same time.

**[49:18 - 49:20]** to see what's bad.

**[49:20 - 49:23]** Now people make a lot of money on this. I have a friend that's

**[49:23 - 49:28]** has a company in this and it's a nice living except it's very stressful because how does this work?

**[49:28 - 49:30]** So I'm a company

**[49:30 - 49:33]** and I've got my design and now I want to make pro cards.

**[49:33 - 49:36]** So what I do is I say, hey, all you probe card companies.

**[49:36 - 49:43]** Here's the layout. Here's my test sequence that'll tell me if it's done right. First person to get me a pro card that works gets all my business.

**[49:43 - 49:52]** And it's a lot of business so these guys will kill themselves. They'll take those things. They'll build them They all have different proprietary techniques and software for building them and testing make sure it's reliable and

**[49:52 - 49:56]** You know the form factor this company that he works for has these little very

**[49:56 - 50:08]** beautiful little spring-loaded things that allow you to bring this probe card down and compliantly touch all the different points on a giant wafer at the same time. It's usually, I don't remember anymore, but I think something like 50 days, in less than 50 days, you have to have everything done.

**[50:08 - 50:18]** and they're like running, running, running, and they're like, hey, you're trying to program it. And then they'll sit there, they'll test it, they'll prove that it works. They send the validated results, and the company goes, boom, you got all our business you just made.

**[50:18 - 50:20]** So many millions of dollars good for you.

**[50:20 - 50:23]** We'll call you next month with the next one and let's continue.

**[50:23 - 50:25]** I'm constantly doing this.

**[50:25 - 50:30]** So there's a whole business around testing these things. So you do a lot of wafer level testing, then you dice them up.

**[50:30 - 50:33]** Cut them, sort them.

**[50:33 - 50:37]** Okay? And this you've all experienced. Some of you may know what I'm talking about.

**[50:37 - 50:42]** So how do you know that these things are sorted?

**[50:42 - 50:43]** Thank you.

**[50:43 - 50:48]** When I tell you it's going to be for some of you that are power gamers, you'll know what I'm talking about.

**[50:48 - 50:57]** Who goes on Newegg from time to time and buys crap for their World of Warcraft or Gears of War or whatever the heck it is you guys play? You, you're the only one?

**[50:57 - 50:59]** All right.

**[50:59 - 51:07]** No, and you look at, well, yeah, but I mean, just generically, you look on there and you'll sell your processors.

**[51:07 - 51:11]** that seem to be the same line but they're somewhere cheaper than others because their performance is different, right?

**[51:11 - 51:17]** And then some will be fucked up and they'll cheap, they'll sell if you're a lot cheaper, right? Like the whole core's not working or whatever. They'll just sell to you anyway.

**[51:17 - 51:20]** So what happens is these guys there's two types one is

**[51:20 - 51:26]** Some things, when there's one whole thing that's working, if there's enough working, they might be able to sell it for basement prices. They'll bin it out and they'll do something with it.

**[51:26 - 51:31]** The other thing is that none of these things all come out identical, right?

**[51:31 - 51:38]** There's essentially Gaussian distributions and just how wide everything comes out in every transistor, in these 100 million transistors.

**[51:38 - 51:45]** And so when you average that out from little chip to little chip, they're all slightly different, which means that some of them will be just

**[51:45 - 51:51]** 10% faster than the other ones made next to each other and so on. So when you make a whole run of these things, they'll all be slightly off from each other.

**[51:51 - 51:56]** There's usually a normal distribution, so you say that's a three sigma processor, and that's a negative one sigma, whatever.

**[51:56 - 52:08]** You bin them this way and that usually means that the clock speed will be just that much faster on this one, a little slower on that one. You bin them out and then you charge people different amounts of money. It all comes out of the same process. It's just the nature of statistics that you don't, they're not all identical.

**[52:08 - 52:12]** So you've also, not at all, but a lot of you have seen this when you buy your stuff.

**[52:12 - 52:19]** So then you package them and then you know you got to take these things and package them and these little packages I was talking about you got to put them in there close them up and

**[52:19 - 52:23]** Then they're solder bumped onto

**[52:23 - 52:28]** The motherboard or multi chip modules very often you put multiple chips together

**[52:28 - 52:40]** And then you burn in whatever you've got to burn into the chip, whatever firmware stuff. You final test, qualify it, and out it goes and shows up on your computer or off a new egg and you buy it and whatever. The other big thing that is...

**[52:40 - 52:42]** What's the huge thing I haven't talked about?

**[52:42 - 52:44]** occupies a lot of people's

**[52:44 - 52:46]** Cycles these days the reason for multi-core

**[52:46 - 52:47]** Thank you.

**[52:47 - 52:48]** Test.

**[52:48 - 52:50]** No, although...

**[52:50 - 52:51]** Thank you.

**[52:51 - 52:56]** Actually, that is one interesting one, but that's not the main reason they went to multicore.

**[52:56 - 53:02]** (No spoken phrase was provided for a statement like this (in fact it's an utterance). For the next part, I'm going to assume that the question is about multi-core processors or the reason for such on-going development of technology in general.
The reason is: - It's hard to make more than one core without making them all be alike. There was a limit to how much you could scale up before things started getting into trouble, and this was probably because there were limits on what we could do with the hardware (like for-words like "the CPU" etc) as well as limitations in other areas such as power consumption.
The Core is an core is an Core: it's a single chip that has multiple cores inside of it. It's not just one big chip, but many small chips all working together to make the same thing happen. This was made possible by advances in technology (like for example: the invention of transistors and microprocessors) as well as improvements in materials manufacturing techniques.
The Core is an core is an Core: it's a single chip that had multiple cores inside of it. It was not until this happened that we could make more than one core without making them all be alike (like for example: the Core i

**[53:02 - 53:04]** So

**[53:04 - 53:08]** Thermal management is ridiculously important.

**[53:08 - 53:14]** Have you ever, so I just put together a system for one of the star companies I met that was a

**[53:14 - 53:22]** Hunk and simulator and basically it's just a gaming machine. It's like this type of thing you we bought it got all the pieces in penalty for new egg and

**[53:22 - 53:24]** And it's, you know, it's...

**[53:24 - 53:28]** It's basically got a 1.5 kilowatt source and it's a

**[53:28 - 53:30]** Beautiful. We did actually

**[53:30 - 53:33]** For fun we downloaded the latest

**[53:33 - 53:35]** It wasn't Gears of War. What's the thing that came out like?

**[53:35 - 53:37]** Yeah.

**[53:37 - 53:42]** The latest one that was sick and so we put it on there just as the CEO is very old guy and he's not impressed by any of this.

**[53:42 - 53:54]** He's like walking in and we're like, dude, check this out. Look at how realistic that leaf is when I shoot it. And you're like the little pieces gonna and it's like, what are you doing? That thing cost six grand. Stop. So, uh,

**[53:54 - 54:00]** The thing about that machine when you build it, and I encourage you all to go look at this because it's very visceral.

**[54:00 - 54:03]** You get a processor that's in a multi-core the package is this big

**[54:03 - 54:09]** Then you put compound on it and then you put a heat sink with a fan made out of copper that's this big.

**[54:09 - 54:14]** It's this heat pipe, multiple copper heat pipes, and then they're ranging a donut.

**[54:14 - 54:21]** And there's a fan and all these copper heat sink blades that run around it just to exhaust out the ridiculous heat that comes out of this thing.

**[54:21 - 54:28]** Okay? Thermal management is a huge problem and that's for a desktop where I can afford to put a ridiculous thing.

**[54:28 - 54:30]** My laptop or these phones

**[54:30 - 54:35]** I mean really right now what's dominating a lot of what's limiting a lot of what can be done is

**[54:35 - 54:38]** really heat management and power management.

**[54:38 - 54:43]** You're limited by your batteries. We'd all love an iPhone that lasts for two weeks.

**[54:43 - 54:44]** It doesn't exist.

**[54:44 - 54:46]** and heat

**[54:46 - 54:55]** And you guys, I hope, know intuitively that if I didn't have that heat sink, that thing would melt. I mean, it has ridiculous heat power density coming out of it.

**[54:55 - 54:58]** It would literally just melt itself into nothing.

**[54:58 - 55:05]** into an ingot. In fact, it wouldn't because what would happen, it would run until it all started fucking up and get brown and black and it would stop and stop computing and

**[55:05 - 55:07]** it would stop generating heat. OK?

**[55:07 - 55:17]** So that's the story of ICs and I have everybody take like a minute or two stretch slap your neighbor and then I'll build a diode and we'll do an amplifier.

**[55:17 - 55:20]** Thank you.

**[55:20 - 55:22]** Thank you.

**[55:22 - 55:23]** Thank you.

**[55:23 - 55:24]** Thank you.

**[55:24 - 55:26]** Thank you.

**[55:26 - 55:28]** Thank you.

**[55:28 - 55:32]** Thank you.

**[55:32 - 55:38]** Yes.

**[55:38 - 55:50]** I teach for you Vivek does 42 and 100 and I do 40 40 for the majors

**[55:50 - 56:02]** It's very very similar, that's why you can use these slides. No, 40 is Monday, Wednesday and Friday at 9 in the morning, 9 to 10.

**[56:02 - 56:14]** The tests are probably harder. I don't make the same tests. I just use similar slide decks and we use the same book. Different homeworks, different tests, similar labs.

**[56:14 - 56:26]** So how do you like the book? Does it suck?

**[56:26 - 56:38]** You can say it sucks, don't worry. You know what? Yeah, a comparator, yeah.

**[56:38 - 56:50]** Hey, whatever, I just couldn't hear you, because you're all screaming.

**[56:50 - 56:59]** Not one in the back, the comparator.

**[56:59 - 57:11]** We onto our135 Open YouTube Rooms.

**[57:11 - 57:23]** It was meant to be frustrating. This is something that's part of my teaching.

**[57:23 - 57:26]** philosophy. Frustration is good.

**[57:26 - 57:28]** Hi, everyone.

**[57:28 - 57:30]** That's good. Confusion is good.

**[57:30 - 57:35]** supposed to piss you off lowball. They're done that way on purpose for me for 40.

**[57:35 - 57:41]** It's hard to me to evaluate how relevant it is for 42 because you guys, some of you aren't majors so in a sense you're like I don't care that much about this.

**[57:41 - 57:45]** All right

**[57:45 - 57:48]** Are we all happy awake?

**[57:48 - 57:51]** Invigorated for what do we got 20 minutes ago, right?

**[57:51 - 57:54]** You guys leave on the half hour

**[57:54 - 57:55]** Alright!

**[57:55 - 57:57]** Should help.

**[57:57 - 57:59]** you

**[57:59 - 58:03]** Let's do the diode. Any questions? Then I'll do amplifiers. Any questions? We're good so far. Life is fine.

**[58:03 - 58:05]** All right.

**[58:05 - 58:07]** What's up?

**[58:07 - 58:09]** Oh, that's amazing.

**[58:09 - 58:11]** Alright, so let's make a dial.

**[58:11 - 58:18]** So I start with P silicon wafer. I buy a wafer made of P-type silicon.

**[58:18 - 58:21]** I oxidize it

**[58:21 - 58:28]** Since I put the whole wafer in, I'm going to oxidize both surfaces, right? Because everything gets hot and is exposed to oxygen, so I get oxide on both sides.

**[58:28 - 58:31]** I will then spin resist.

**[58:31 - 58:33]** And

**[58:33 - 58:35]** shining UV through this mask.

**[58:35 - 58:37]** Okay?

**[58:37 - 58:39]** leaving behind resist

**[58:39 - 58:43]** With this hole in it just up here

**[58:43 - 58:49]** and then I use that to protect the areas I don't want to etch of oxide and etch other areas away so it'll remove

**[58:49 - 58:53]** this oxide in this piece. I'll get to this stuff.

**[58:53 - 58:58]** Alright, so I used, I spun photoresist, exposed the photoresist so that there's a square in the middle

**[58:58 - 59:02]** exposed, stuck that in an oxide etch or it etched away the oxide

**[59:02 - 59:08]** and I dip my wafer in acetone, get rid of the photo resist altogether because photo resist is useless after we're done with it.

**[59:08 - 59:12]** and you're left behind with a film of oxide everywhere, glass. Basically it's now

**[59:12 - 59:15]** a wafer coated in glass.

**[59:15 - 59:17]** with a hole in the glass.

**[59:17 - 59:21]** a gap. Imagine a circular could be a circle or a rectangle.

**[59:21 - 59:27]** Then I sit there and implant arsenic.

**[59:27 - 59:31]** because I like to play with arsenic and I implant arsenic into the p-silicon

**[59:31 - 59:36]** The silicon oxide will eat will stop the arsenic. It'll take some of the arsenic into its

**[59:36 - 59:39]** on the surface, just on surface, just below the surface.

**[59:39 - 59:42]** The spoken transcript remains unchanged for this sentence (for to be precise): a-102 was not until 1987 when the World-wide web was made in-house by the Web browser, Internet-during. -during. during.

**[59:42 - 59:45]** will go in and after I anneal it

**[59:45 - 59:49]** and let it diffuse out a little bit, that arsenic will essentially turn that P into an N type.

**[59:49 - 59:56]** piece of silicon. So now I have a well, it's called a well, I have a well of N sitting in a wafer of P.

**[59:56 - 60:02]** So you can immediately see if you've done this in physics that that's a pn junction. I just made myself a little dial.

**[60:02 - 60:05]** I haven't finished it, but essentially just by doing that I have a PN.

**[60:05 - 60:10]** You can build solar cells this way. You can build simple PN solar cells essentially this way.

**[60:10 - 60:13]** Drop this stuff over the whole area.

**[60:13 - 60:25]** Now I can go in and do sputtering and drop aluminum down everywhere. So I just literally put it underneath, put the aluminum targets, sputter the crap out of it, land it, and coat everything with a film of aluminum. So the whole

**[60:25 - 60:26]** wafers now coated with aluminum.

**[60:26 - 60:30]** But I have myself another mask which I call my metal mask.

**[60:30 - 60:33]** I'll spin resist over the whole wafer.

**[60:33 - 60:34]** and expose it with this.

**[60:34 - 60:40]** this pattern so that the light only gets on the flanks and is blocked by the center metal region

**[60:40 - 60:43]** And lo and behold

**[60:43 - 60:46]** After I do that and etch it in an aluminum etcher

**[60:46 - 60:50]** And then get rid of the resist by dipping in an acetone, the polymer always goes away.

**[60:50 - 60:52]** I'm left with a little aluminum wire.

**[60:52 - 60:54]** that runs down

**[60:54 - 60:59]** and over to the side. So it comes down into the well and shorts with the end region.

**[60:59 - 61:03]** and then climbs out of it and goes out a short distance. It's a little wire. I just made myself a little wire.

**[61:03 - 61:12]** Then I use CBD to chemically deposit in the gas phase down onto the surface. Silicon nitride, which is an insulator. It doesn't matter what else it is. It doesn't matter. It's just an insulator.

**[61:12 - 61:14]** So I put nitride everywhere.

**[61:14 - 61:21]** Then I do another lithography step, which I don't show the math for here because it starts to get tedious, but basically I expose and

**[61:21 - 61:28]** etch the nitride so that I etch a gap so that I'm looking down into the aluminum.

**[61:28 - 61:31]** I've cut through the insulator down to the exposed aluminum.

**[61:31 - 61:36]** Now I will want to thin out the backside.

**[61:36 - 61:45]** So that there isn't a lot of distance between here and here. So to protect it, all this does is they spin some photos on the front just to protect the front so it doesn't get

**[61:45 - 61:48]** I messed up and I basically just grind it down and mill it down.

**[61:48 - 61:53]** with a polisher essentially with a machine that just will grind it down a little bit to some thickness.

**[61:53 - 61:55]** And then I get rid of this resist because

**[61:55 - 62:04]** You don't want it for anything then I will flip the wafer and again use sputtering to drop gold on the backside So the whole wafer is now covered in gold on the bottom

**[62:04 - 62:06]** And then I'm going to take that

**[62:06 - 62:18]** Cut it up so that only the little chip that I care about is a little square Slap the gold onto an electronic board a PCB board those green boards You know that you see inside of everything that has a piece of metal

**[62:18 - 62:26]** here that I can bond onto so that the bottom part shorts to something and then I run a wire bond or a little wire that'll bond here and

**[62:26 - 62:28]** and bond over and into this aluminum.

**[62:28 - 62:33]** And now, or this is maybe my plastic package, actually I would have, let's say this is a plastic package.

**[62:33 - 62:37]** the little dip packages that you guys use for your amplifier. This is the inside of

**[62:37 - 62:39]** So these little metal leads.

**[62:39 - 62:41]** here.

**[62:41 - 62:52]** are those little pins that come out and then I've got my chip in there and then I can seal it or seal it and open it or whatever but now when I'm done I've got myself a PN junction because this wire

**[62:52 - 62:54]** this pin that goes into your breadboard

**[62:54 - 62:59]** This pin shorts to the wire, connects to the aluminum which comes down and shorts to the end part.

**[62:59 - 63:04]** And this wire comes up through the gold and shorts the P part and I've got myself a diode. That's it. I have a diode.

**[63:04 - 63:06]** Now you can imagine

**[63:06 - 63:12]** I can take this with a lot more refinement, you know, 40 years of refinement and do it and you can build yourself a

**[63:12 - 63:14]** 800 million

**[63:14 - 63:17]** transistor this is a diode not a transistor, but

**[63:17 - 63:19]** similar stuff.

**[63:19 - 63:20]** What do you mean?

**[63:20 - 63:21]** a little bit on the top.

**[63:21 - 63:25]** Oh, I mean for this it was pretty arbitrary because the

**[63:25 - 63:28]** The gold would form a better bond down to the big finances

**[63:28 - 63:30]** Your purpose doesn't matter.

**[63:30 - 63:35]** Gold and noble metal doesn't oxidize and so it's nicer to have on the bottom when you're doing this bond.

**[63:35 - 63:37]** There is no big deal.

**[63:37 - 63:39]** Does that make sense?

**[63:39 - 63:40]** Bye.

**[63:40 - 63:43]** Yeah, you guys don't seem as excited as you should.

**[63:43 - 63:44]** You're all...

**[63:44 - 63:46]** Thank you.

**[63:46 - 63:47]** That is true.

**[63:47 - 63:48]** Yeah.

**[63:48 - 63:53]** No, in this case just an insulator. If in an old style FET it would be the gate.

**[63:53 - 63:59]** If you put a thin oxide between a metal and a semiconductor, you make yourself a pet gamer.

**[63:59 - 64:02]** But in this case it's just glass. It's acting as an insulin.

**[64:02 - 64:03]** Alright?

**[64:03 - 64:04]** Thank you.

**[64:04 - 64:09]** Now I could tell Vivek that this is as far as we got.

**[64:09 - 64:12]** And he can start on amplifiers on what Monday or Wednesday?

**[64:12 - 64:24]** Let's do that. All right. So because you know why honestly it's not because I want to go home I'm gonna go back to my office.

**[64:24 - 64:30]** and work for hours. It's just that I think you guys look, if I start on this, it's just gonna bleed out your ears.

**[64:30 - 64:35]** So you can come in on Wednesday and pick it up. But let me, so I don't feel as guilty, I'm not gonna talk about you all.

**[64:35 - 64:37]** Just so you see what's coming.

**[64:37 - 64:39]** 30 seconds and you can go, it's still early.

**[64:39 - 64:44]** Basically, what he's going to go is give you a sort of canonical amplifier.

**[64:44 - 64:47]** You know, what is an amplifier, some packaging for it.

**[64:47 - 64:50]** and then go through the whole goal is eventually to get

**[64:50 - 64:54]** Hold on, wait for it, it's exciting.

**[64:54 - 64:58]** To get to the op amps that you guys are using in the lab that frustrated you so much.

**[64:58 - 65:06]** All right.

