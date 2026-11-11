"""Create a reproducible synthetic bike-share maintenance dataset."""

import argparse
import csv
import random
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "notes.csv"

# Each complaint is paired with several neutral contexts in training.
# Validation and test notes are authored separately and never use these wrappers.
TRAIN = {
    "brake": [
        "the front brake lever reaches the handlebar",
        "the rear brake barely slows the bike",
        "the brake pads scrape the rim",
        "the front brake makes a loud squeal",
        "the left lever feels spongy when I squeeze it",
        "the brake cable looks frayed by the lever",
        "the bike takes too long to stop",
        "the rear wheel locks as soon as I brake",
        "the right brake handle sticks after release",
        "the brakes pulse when I slow down",
        "the brake shoes are rubbing on one side",
        "the front caliper is loose",
        "the brake lever has no resistance",
        "there is a grinding noise when stopping",
        "the pads look worn down to the metal",
        "the front brake grabs suddenly",
        "the brake cable snapped during the ride",
        "the rear brake drags even with the lever released",
    ],
    "drivetrain": [
        "the chain falls off when I pedal",
        "the gears jump under pressure",
        "the shifter will not move into low gear",
        "the pedals spin without moving the bike",
        "the chain skips over the rear sprocket",
        "the crank clicks on every turn",
        "the derailleur is bent toward the spokes",
        "the chain is jammed between the gears",
        "the gear cable came loose",
        "the pedals feel rough under load",
        "the chain rattles against the guard",
        "the bike will not shift to a higher gear",
        "the right crank arm wobbles",
        "the chain has a stiff rusty link",
        "the shifter changes two gears at a time",
        "the cassette slips when climbing",
        "the pedal threads feel loose",
        "the chain jumped off the front ring",
    ],
    "tire": [
        "the front tire is completely flat",
        "the rear tire loses air overnight",
        "the tube has a puncture",
        "the valve leaks after inflation",
        "the tire has a cut in the sidewall",
        "the rear tire feels soft on bumps",
        "the tire bead is coming off the rim",
        "there is glass stuck in the tread",
        "the front tire has a bulge near the rim",
        "the tire deflates as soon as I remove the pump",
        "the valve stem is bent",
        "the rear wheel bounces because the tire is uneven",
        "the tube keeps going flat",
        "the tread has a deep split",
        "the front tire hisses when pressed",
        "the valve cap is off and the stem leaks",
        "the tire will not hold pressure",
        "a nail went through the rear tire",
    ],
    "lock": [
        "the dock will not release the bike",
        "the latch sticks when I return the bike",
        "the bike is stuck in the station",
        "the dock light stays red after I push the bike in",
        "the locking pin will not retract",
        "the app says unlocked but the bike is still held",
        "the return slot will not latch",
        "the lock bolt is jammed",
        "the bike clicks into the dock but can still be pulled out",
        "the station never confirms that the bike is returned",
        "the lock does not open after scanning",
        "the docking handle is stuck halfway",
        "the dock releases the bike then immediately locks again",
        "the lock housing looks bent",
        "the rental ends but the lock has not engaged",
        "the station rejects the bike at every return attempt",
        "the docking pin does not line up with the slot",
        "the dock holds the bike even after a green light",
    ],
}

CONTEXTS = [
    "At the station I noticed {issue}.",
    "Bike {number}, {issue}.",
    "Please check bike {number} because {issue}.",
    "Flagging this after my morning ride, {issue}.",
    "Can someone look at bike {number}? {issue}.",
    "I tried this bike today and {issue}.",
    "Maintenance note for bike {number}. {issue}.",
    "On the way home I found that {issue}.",
    "The last rider reported that {issue}.",
    "Leaving this for the next mechanic, {issue}.",
    "Quick note from the afternoon shift, {issue}.",
    "I stopped using bike {number} because {issue}.",
]

VALIDATION = {
    "brake": [
        "It only slows if I squeeze both levers hard.",
        "Something screeches every time I try to stop.",
        "The rear brake does not spring back.",
        "The stopping distance feels much longer today.",
        "A strand is sticking out of the front brake cable.",
        "The brake handle bends almost to the grip.",
    ],
    "drivetrain": [
        "My foot drops suddenly every time the chain jumps.",
        "Turning the shifter makes no difference uphill.",
        "There is a clunk from the crank when I stand to pedal.",
        "The rear gears skip on the middle setting.",
        "The chain is wedged under the plastic cover.",
        "The pedal on the left side feels like it could fall off.",
    ],
    "tire": [
        "The rubber is soft again a minute after pumping.",
        "A small thorn went into the front tread.",
        "The rear tube leaks near the valve.",
        "The tire peeled away from the edge of the wheel.",
        "There is a lump on the side of the front tire.",
        "The rear wheel is almost rolling on the rim.",
    ],
    "lock": [
        "The bike stays attached even when the ride starts.",
        "I returned it but the slot never caught the bike.",
        "The docking latch is slow to open.",
        "My trip timer kept running after the bike went into the station.",
        "The dock pin is frozen in place.",
        "It was possible to pull out the bike without starting a ride.",
    ],
}

TEST = {
    "brake": [
        "Pulled the lever to the grip and the bike kept rolling downhill.",
        "Metal scraping only when I use the rear brake.",
        "The front wheel stops with a jerk as soon as I squeeze.",
        "Left handle stays squeezed unless I push it back.",
        "Had to put both feet down to stop at the crossing.",
        "There is oil on the rotor and the stopping power is poor.",
        "Rear pads rub constantly after I let go of the lever.",
        "The right lever suddenly went slack at the intersection.",
        "Sharp squeal from the front wheel while slowing down.",
        "One pad is touching the rim even while I coast.",
        "Braking feels weak in the first few seconds of the ride.",
        "I can squeeze the handle all the way with barely any slowing.",
    ],
    "drivetrain": [
        "A hard pedal stroke makes the chain jump a tooth.",
        "The right pedal turns but the back wheel gets no power.",
        "I can hear the chain hitting the cover with each turn.",
        "It feels stuck in the hardest gear even after I twist the shifter.",
        "The crank moves sideways every time I stand on the pedals.",
        "One chain link stays bent as it passes over the gear.",
        "The rear gear selector is hanging too close to the spokes.",
        "The pedal came off when I started uphill.",
        "It changes gears by itself on the climb.",
        "The bike loses drive for a moment whenever I push hard.",
        "Chain came loose when I shifted at the light.",
        "The shifter clicks but the chain stays on the same cog.",
    ],
    "tire": [
        "The front wheel is squishy and there is a pin in the rubber.",
        "Air escapes from the valve as soon as the pump disconnects.",
        "A sidewall bubble has formed on the back wheel.",
        "The rim hits the road when I sit down because the tire is so soft.",
        "The back tire slowly goes down between trips.",
        "A piece of the tire edge is outside the rim.",
        "I can see the tube through a gash in the side.",
        "The tread has a piece of wire pushed into it.",
        "There is a steady hiss from the rubber around the front wheel.",
        "The rear tire is low again after I topped it up.",
        "It felt bumpy and I found a cut across the tread.",
        "The valve stem wiggles and air escapes at the base.",
    ],
    "lock": [
        "The rental started but the front wheel stayed trapped in the dock.",
        "I pushed the bike all the way in and the latch never closed.",
        "The station flashed green but would not let go.",
        "End ride still shows a warning even after I docked the bike.",
        "The locking piece is sticking out too far to return the bike.",
        "The bike pulls free even though the station shows it as secure.",
        "Scanning worked but the slot is holding the bike.",
        "The station pin moved halfway and got stuck during pickup.",
        "The empty slot keeps rejecting the front wheel at return.",
        "The ride is still running and the dock made no confirmation sound.",
        "The lock engaged for a second and then let the bike slip back out.",
        "I hear a click on pickup but the bike will not come free.",
    ],
}

ABBREVIATIONS = {
    "and": "n", "before": "b4", "because": "bc", "with": "w",
    "without": "wout", "front": "frnt", "rear": "rr", "wheel": "whl",
    "right": "rt", "left": "lft", "little": "lil", "when": "wen",
    "today": "tdy", "station": "stn", "seconds": "secs", "though": "tho",
}


def noisy_version(note, rng):
    words = re.findall(r"[a-z0-9]+", note.lower())
    words = [ABBREVIATIONS.get(word, word) for word in words]
    # Swap adjacent interior letters in roughly one out of five eligible words.
    eligible = [
        index for index, word in enumerate(words)
        if any(word[i] != word[i + 1] for i in range(1, len(word) - 2))
    ]
    for index in rng.sample(eligible, max(1, round(len(eligible) * 0.2)) if eligible else 0):
        word = words[index]
        choices = [i for i in range(1, len(word) - 2) if word[i] != word[i + 1]]
        i = rng.choice(choices)
        words[index] = word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return " ".join(words)


def build_dataset(path=DATA):
    rng = random.Random(42)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "split", "label", "text", "noisy_text"])
        writer.writeheader()

        for label, complaints in TRAIN.items():
            for complaint_index, complaint in enumerate(complaints, start=1):
                for context_index, context in enumerate(CONTEXTS, start=1):
                    writer.writerow({
                        "id": f"train-{label}-{complaint_index}-{context_index}",
                        "split": "train",
                        "label": label,
                        "text": context.format(issue=complaint, number=rng.randint(100, 999)),
                        "noisy_text": "",
                    })

        for split, notes_by_label in [("validation", VALIDATION), ("test", TEST)]:
            for label, notes in notes_by_label.items():
                for index, note in enumerate(notes, start=1):
                    writer.writerow({
                        "id": f"{split}-{label}-{index}",
                        "split": split,
                        "label": label,
                        "text": note,
                        "noisy_text": noisy_version(note, rng) if split == "test" else "",
                    })

    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA)
    args = parser.parse_args()
    print(f"Wrote {build_dataset(args.output)}")
