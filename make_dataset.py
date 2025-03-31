import librosa
import numpy as np, scipy as sp
from scipy.signal import windows, minimum_phase
from scipy import stats
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from sklearn.preprocessing import minmax_scale
from random import randint
import skimage.measure as skimeasure
from scipy.interpolate import PchipInterpolator

np.seterr(all="raise")
np.seterr(under="ignore")
from scipy.special import seterr

seterr(all="raise")
seterr(underflow="warn")

import microexperiment_random_resonances_fr as rando_fr

from sys import path

path.append("/Users/[redacted]//")
import utils, consts_windows, resonot_adapter, overlap_add_2

path.append("/Users/[redacted]///")
import harsh_speaker_filter

SR = 48000

DB_DYNAMIC_RANGE = 100

DUR_TIME_CROPPED_S = 1
DUR_TIME_OVERSCAN_S = 1.25

DIMS_CROPPED = [(2 * 64, 256), (2 * 32, 512), (2 * 16, 1024), (2 * 32, 1024)]
DIMS_OVERSCAN = np.array(
    [(_x * DUR_TIME_OVERSCAN_S / DUR_TIME_CROPPED_S, _y) for _x, _y in DIMS_CROPPED]
)
_DIMS_OVERSCAN_int = DIMS_OVERSCAN.astype(int)
assert np.all((DIMS_OVERSCAN - _DIMS_OVERSCAN_int) == 0)
DIMS_OVERSCAN = _DIMS_OVERSCAN_int

PROPORTION_IN_DATASET__ARITFACTS = 55 / 100


def rms(signal):
    return np.sqrt(np.mean(signal**2))


def add_white_noise(signal, vol_dB=-120):
    return signal + np.random.normal(0, librosa.db_to_amplitude(vol_dB), signal.shape)


def preprocess_source(signal, sr):
    signal_filtered = harsh_speaker_filter.apply_mobile_device_speaker_fr(
        signal[np.newaxis, :], sr
    )[0]
    signal_normalized = signal_filtered / rms(signal_filtered)
    signal_noised = add_white_noise(signal_normalized)
    return signal_noised


POLARITIES = np.array([-1.0, 1.0], dtype=np.float64)


def fetch_chunk(
    signal,
    l_sam,
    use_loc=None,
    augmentation_src_sig=None,
    augmentation_src_sig_prefiltered=None,
    for_training=True,
):
    if for_training:
        assert l_sam == SR * DUR_TIME_OVERSCAN_S
    CHANCE_AUGMENT_ENVIRONMENT_NOISE = 0.15 if for_training else 0
    loc = (
        int(round(np.random.uniform(0, len(signal) - l_sam)))
        if use_loc is None
        else use_loc
    )
    chunk = signal[loc : loc + l_sam]
    _crop = (l_sam - SR * DUR_TIME_CROPPED_S) / 2
    assert utils.is_an_int(_crop)
    _crop = int(_crop)
    rms_chunk = rms(chunk[_crop : len(chunk) - _crop] if for_training else chunk)
    chunk_normalized = chunk / rms_chunk if rms_chunk > 0 else chunk
    chunk_noised = (
        add_white_noise(chunk_normalized)
        if (
            augmentation_src_sig is None
            or np.random.uniform(0, 1) >= CHANCE_AUGMENT_ENVIRONMENT_NOISE
        )
        else (
            chunk_normalized
            + fetch_chunk(
                (
                    augmentation_src_sig_prefiltered
                    if np.random.uniform(0, 1) <= 2 / 3
                    else augmentation_src_sig
                ),
                l_sam,
            )
            * librosa.db_to_amplitude(np.random.uniform(-50, -12.5))
            * np.random.choice(POLARITIES)
        )
    )
    """
    if use_loc is None and np.random.uniform(0, 1) < CHANCE_AUGMENT_BACKWARDS:
        chunk_noised = chunk_noised[::-1]  # backwards
    """
    # flipping across time axis is now done in Kaggle to 1 in 4 original train items
    return chunk_noised / rms(chunk_noised)


def dB_hearing_scale_softplus(dB_raw):
    a = 0.1
    return np.log(1 + np.exp(a * (DB_DYNAMIC_RANGE + dB_raw))) / a


def is_a_power_of_2(n):
    return (n & (n - 1) == 0) and n != 0


def signal_to_spectrogram(
    signal,
    dims: tuple,
    l_win_fft: int,
    is_for_training: bool = True,
    oversample: int = 3,
):
    hop = SR * DUR_TIME_OVERSCAN_S / dims[0] / oversample
    assert is_a_power_of_2(dims[1]) and utils.is_an_int(hop)
    hop = int(hop)
    # assert utils.is_an_int(hop / 2)
    spectrogram = (
        consts_windows.RMS_FLATTOP
        * np.abs(
            librosa.stft(
                signal[hop // 2 :],  # np.pad(signal, (hop // 2, hop)),
                n_fft=l_win_fft,
                hop_length=hop,
                window=windows.flattop,
            )
        )
        / dims[1]
    )
    spectrogram = np.swapaxes(spectrogram, 0, 1)
    oversamp_remaineder = len(spectrogram) % oversample
    if is_for_training:
        assert oversamp_remaineder == 0
    elif oversamp_remaineder > 0:
        infill = oversample - oversamp_remaineder
        spectrogram = np.concatenate(
            (spectrogram, np.repeat(spectrogram[-1:], repeats=infill, axis=0)), axis=0
        )
    spectrogram = np.sqrt(
        skimeasure.block_reduce(spectrogram**2, (oversample, 1), np.mean)
    )
    if is_for_training:
        if spectrogram.shape[0] == dims[0] + 1:
            spectrogram = spectrogram[:-1]
        # print(spectrogram.shape[0], dims[0])
        assert spectrogram.shape[0] == dims[0]
    assert spectrogram.shape[1] == 1 + l_win_fft // 2

    return spectrogram


def enum(names):
    return {n: i for i, n in enumerate(names)}


ENUM_ARTIFACTS = enum(
    [
        "sines",
        "resonances",
        "blips",
        "fr-jumps",
        "sampled-harshness",
        "comb",
        "random-spiky-fr",
    ]
)


def alter_chunk(signal, sr, aug_harsh_signalbank=None):

    resonances_filters_mode = np.random.choice([1, 2, 3], p=[0.3, 0.6, 0.1])

    def bell_curve(x, x_mode, Q, fat_tails=False):
        x_transformed = Q * (x - x_mode)
        return np.exp(
            -(
                (
                    np.arctan(np.pi * x_transformed) * np.sqrt(np.abs(x) / 2)
                    if fat_tails
                    else x_transformed
                )
                ** 2
            )
        )

    def step_bandpass(x, a):
        s = 1 / a
        tapered_cauchy = lambda x: (1 / (1 + x**2)) * (np.exp(-((x / 25) ** 2)))
        return np.cbrt(2 * tapered_cauchy(s * x) * s * x)

    l = len(signal)
    l_s = l / sr
    indices_signal_samples = np.arange(l)
    f_Hz_nyquist = sr / 2

    fr_x_Hz_per_bin = f_Hz_nyquist * np.linspace(1 / 4096, 4095 / 4096, 4095)

    def get_sine(_f, _amp):
        return _amp * np.sin(_f * indices_signal_samples * (2 * np.pi) / sr)

    def get_resonated():
        fr_dB = np.zeros((4095,), dtype=np.float64)
        octs = utils.hertz_to_C_octaves(fr_x_Hz_per_bin)

        use_fat_tail_bells = np.random.uniform(0, 1) <= 1 / 2

        gain_dB_low_bound = utils.map_range(n_resonances, (1, 10), (-1.25, -2.5))
        gain_dB_high_bound = utils.map_range(n_resonances, (1, 10), (30, 20))

        GLOBAL_NEGATIVE_GAIN_MAGNIT_LIMIT_DB = 1.25

        for i in range(
            int(
                round(
                    max(
                        np.random.uniform(1, 1 + n_resonances),
                        np.random.uniform(1, 1 + n_resonances),
                    )
                )
            )
            if do_dynamic_resonances
            else n_resonances
        ):
            f = np.clip(
                (
                    (6000 * 2 ** np.random.normal(0, 1.618))
                    if (np.random.uniform(0, 1) <= 3 / 4)
                    else (250 * 80 ** np.random.uniform(0, 1))
                ),
                250,
                f_Hz_nyquist,
            )  # octaves
            q = (
                (1.618 if use_fat_tail_bells else 1)
                * [
                    np.random.uniform(6.25, 12.5),
                    np.random.uniform(12.5, 25),
                    np.random.uniform(25, 50),
                    np.random.uniform(50, utils.map_range(f, (2500, 20000), (75, 250))),
                ][np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])]
                # if gain > 0
                # else np.random.uniform(3.75, 10)
            )
            gain = np.random.uniform(
                gain_dB_low_bound,
                gain_dB_high_bound
                + utils.map_range(np.log2(q), (np.log2(25), np.log2(100)), (0, 10)),
            )
            fr_dB += (
                gain
                * bell_curve(octs, utils.hertz_to_C_octaves(f), q, use_fat_tail_bells)
                * (
                    bell_curve(
                        octs, utils.hertz_to_C_octaves(f), q / np.pi, fat_tails=False
                    )
                    if use_fat_tail_bells
                    else 1
                )
            )

        fr_dB *= utils.smootherstep(
            utils.map_range_vector(
                octs,
                (utils.hertz_to_C_octaves(500), utils.hertz_to_C_octaves(1500)),
                (0, 1),
            )
        )
        where_eq_subzero = fr_dB[fr_dB < 0,]
        tallest_dip_bell_height_dB = (
            np.max(np.abs(where_eq_subzero)) if len(where_eq_subzero) else 0
        )
        if tallest_dip_bell_height_dB > GLOBAL_NEGATIVE_GAIN_MAGNIT_LIMIT_DB:
            fr_dB[fr_dB < 0,] /= (
                tallest_dip_bell_height_dB / GLOBAL_NEGATIVE_GAIN_MAGNIT_LIMIT_DB
            )

        do_shifting_resonances = np.random.uniform(0, 1) < 1 / 5

        if not do_shifting_resonances:
            ir = utils.make_ir(
                librosa.db_to_amplitude(fr_dB)
                ** (1 if resonances_filters_mode == 2 else 2),
                1 / 10,
                sr,
            )
            if resonances_filters_mode != 2:
                ir = utils.ir_to_minimum_phase(ir)
                if resonances_filters_mode == 3:
                    ir = ir[::-1]

            signal_resonated = utils.center_kernel_convolve_segment(
                0, l, signal_copy, ir
            )
            return signal_resonated

        LEN_GRAIN_SAM = 2400
        LEN_HOP_SAM = 1200
        manager_grains = overlap_add_2.BlockProcessManager(
            signal_copy, LEN_GRAIN_SAM, 2
        )
        # assert np.all(np.isfinite(fr_x_Hz_per_bin))
        # assert np.all(np.isfinite(fr_dB))
        try:
            interpolator_fr_dB = PchipInterpolator(
                *utils.const_extrap(fr_x_Hz_per_bin, 100.0 + fr_dB), extrapolate=True
            )
        except Exception:
            # print(fr_x_Hz_per_bin.tolist())
            print(fr_dB.tolist())
        blocks_iterable = sorted(
            manager_grains.blocks_iterable(), key=lambda g: g.bounds[0]
        )
        fr_pitch_shift_per_block = np.random.uniform(0.0, 1.0, len(blocks_iterable))
        fr_pitch_shift_per_block = utils.fast_blurfilter(
            fr_pitch_shift_per_block,
            2.0 ** np.random.uniform(-4.0, 0.0) * sr / LEN_HOP_SAM,
        )
        fr_pitch_shift_per_block = minmax_scale(fr_pitch_shift_per_block, (0.0, 1.0))
        fr_pitch_shift_per_block = (
            utils.inverse_smoothstep(fr_pitch_shift_per_block) - 1.0
        )
        fr_pitch_shift_per_block = (2.0 ** (0.5 / 12.0)) ** fr_pitch_shift_per_block
        for shift_pitch_this_block, subsig in zip(
            fr_pitch_shift_per_block, blocks_iterable
        ):
            # print(shift_pitch_this_block)
            ir = utils.make_ir(
                librosa.db_to_amplitude(
                    interpolator_fr_dB(shift_pitch_this_block * fr_x_Hz_per_bin) - 100.0
                ),
                1 / 15,
                sr,
            )
            subsig.set(subsig.convolve(ir))
        manager_grains.write_to_base()
        return manager_grains.base

    # 1 = only sines | 2 = only resonances | 3 = only blips | 4 = sines and resonances | 5 = resonances and blips | 6 = sines and blips | 7 = all
    # mode = randint(1, 7)

    artifacts_to_add = [
        ENUM_ARTIFACTS["sines"],
        ENUM_ARTIFACTS["resonances"],
        ENUM_ARTIFACTS["blips"],
        ENUM_ARTIFACTS["fr-jumps"],
        ENUM_ARTIFACTS["sampled-harshness"],
        ENUM_ARTIFACTS["comb"],
        ENUM_ARTIFACTS["random-spiky-fr"],
    ]
    artifacts_probabilities = [
        0.175,
        0.3625,
        0.225,
        0.1,
        0.025,
        0.075,
        0.0375,
    ]
    n_artifacts_to_add = np.random.choice(
        [1, 2, 3, 4, 5, 6, 7], p=[0.4, 0.29, 0.15, 0.08, 0.05, 0.02, 0.01]
    )
    artifacts_to_add_picked_indices = np.random.choice(
        np.arange(len(artifacts_to_add)),
        size=n_artifacts_to_add,
        replace=False,
        p=artifacts_probabilities,
    )
    artifacts_picked = [artifacts_to_add[_i] for _i in artifacts_to_add_picked_indices]

    n_sines = (
        int(np.clip(np.random.uniform(1, np.random.pareto(0.5) + 2.5), 1, 10))
        if ENUM_ARTIFACTS["sines"] in artifacts_picked
        else 0
    )
    n_resonances = (
        int(np.clip(np.random.uniform(1, np.random.pareto(0.5) + 5), 1, 25))
        if ENUM_ARTIFACTS["resonances"] in artifacts_picked
        else 0
    )
    n_blips = (
        np.random.choice([1, 2, 3, 4, 5], p=[0.45, 0.25, 0.15, 0.1, 0.05])
        if ENUM_ARTIFACTS["blips"] in artifacts_picked
        else 0
    )
    n_unsmooths = (
        np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        if ENUM_ARTIFACTS["fr-jumps"] in artifacts_picked
        else 0
    )

    signal_copy = np.copy(signal)

    if ENUM_ARTIFACTS["random-spiky-fr"] in artifacts_picked:
        spikes_l_grain_sam = np.random.choice(
            [1 / 8, 1 / 4, 1 / 2, 1], p=[0.1, 0.3, 0.3, 0.3]
        )
        if spikes_l_grain_sam == 1:
            spikes_fr = librosa.db_to_amplitude(
                rando_fr.get() * np.sqrt(np.random.uniform(0, 1))
            )
            spikes_ir = utils.make_ir(spikes_fr[1:], 1 / 10, sr)
            signal_copy = utils.center_kernel_convolve_segment(
                0, l, signal_copy, spikes_ir
            )
        else:
            spikes_l_grain_sam *= sr
            assert utils.is_an_int(spikes_l_grain_sam)
            spikes_l_grain_sam = int(spikes_l_grain_sam)

            spikes_manager_grains = overlap_add_2.BlockProcessManager(
                signal_copy,
                spikes_l_grain_sam,
                2 if np.random.uniform(0, 1) < (1 / 2) else 3,
            )
            for spikes_subsig in spikes_manager_grains.blocks_iterable():
                spikes_fr = librosa.db_to_amplitude(
                    rando_fr.get() * np.sqrt(np.random.uniform(0, 1))
                )
                spikes_ir = utils.make_ir(spikes_fr[1:], 1 / 10, sr)
                spikes_subsig.set(spikes_subsig.convolve(spikes_ir))
            spikes_manager_grains.write_to_base()
            signal_copy = spikes_manager_grains.base

    if n_sines > 0:

        for i in range(n_sines):
            f = np.random.uniform(5000, 24000)
            amp = librosa.db_to_amplitude(np.random.uniform(-87.5, -37.5))
            signal_copy += get_sine(f, amp)

    if n_resonances > 0:

        do_dynamic_resonances = np.random.uniform(0, 1) <= 1 / 2  # 3 / 7

        if do_dynamic_resonances:
            resonateds = (get_resonated(), get_resonated())
            which = minmax_scale(
                utils.superfast_blurfilter(
                    np.random.uniform(0, 1, l), np.random.uniform(sr / 10, sr)
                ),
                (np.random.uniform(0, 1 / 3), np.random.uniform(2 / 3, 1)),
            )
            howmuch = minmax_scale(
                utils.superfast_blurfilter(
                    np.random.uniform(0, 1, l), np.random.uniform(sr / 5, sr)
                ),
                (np.random.uniform(0, 1 / 4), 1),
            )
            howmuch = utils.map_range_vector(
                np.random.uniform(0, 1), (0, 1), (howmuch, utils.smootherstep(howmuch))
            )
            which = utils.map_range_vector(
                np.random.uniform(0, 1), (0, 1), (which, utils.smootherstep(which))
            )
            signal_copy = utils.map_range_vector(
                howmuch,
                (0, 1),
                (signal_copy, utils.map_range_vector(which, (0, 1), resonateds)),
            )

        else:
            signal_copy = get_resonated()

    if n_blips > 0:

        # uses spectrogram to make blips more likely to be added at times and frequencies that are already present
        # unlikely for random blip to be unrelated to the rest of the signal
        gram = trimmed_spectrogram(signal, 32, 4096, 512)
        t_per_gram_cell = np.tile(
            np.arange(gram.shape[0], dtype=np.float64), (gram.shape[1], 1)
        ).T
        t_per_gram_cell += np.random.uniform(0.0, 1.0, size=t_per_gram_cell.shape)
        t_per_gram_cell = t_per_gram_cell / gram.shape[0] * l
        f_per_gram_cell = np.tile(
            np.arange(1, 1 + gram.shape[1], dtype=np.float64), (gram.shape[0], 1)
        ) / (1 + gram.shape[1])
        f_per_gram_cell += np.random.uniform(0.0, 1.0, size=f_per_gram_cell.shape)
        f_per_gram_cell = f_per_gram_cell * sr / 2
        gram_cells_flat = gram.flatten()
        gram_cells_flat_indices = np.arange(gram.flatten().shape[0])
        t_per_gram_cell_flat = t_per_gram_cell.flatten()
        f_per_gram_cell_flat = f_per_gram_cell.flatten()
        gram_cells_flat_probabilities = gram_cells_flat**2
        gram_cells_flat_probabilities /= np.sum(gram_cells_flat_probabilities)

        for i in range(n_blips):
            index_picked = np.random.choice(
                gram_cells_flat_indices, p=gram_cells_flat_probabilities
            )
            f = f_per_gram_cell_flat[
                index_picked
            ]  # 100 * 240 ** np.random.uniform(0, 1)
            amp = librosa.db_to_amplitude(
                gram_cells_flat[index_picked]
                - DB_DYNAMIC_RANGE
                + np.random.uniform(+12.5, +37.5)
            )
            wave = get_sine(f, amp)
            loc_blip = int(
                t_per_gram_cell_flat[index_picked]
            )  # np.random.uniform(0, l)
            """
            blip_is_short = np.random.uniform(0, 1) <= 1 / 2
            sharpness_blip = (
                f * np.random.uniform(0.5, 1.5)
                if blip_is_short
                else (
                    np.random.uniform(50, 25)
                    if np.random.uniform(0, 1) <= 2 / 3
                    else np.random.uniform(25, 1 / 10)
                )
            ) / sr
            """
            sharpness_blip = (
                [
                    f * np.random.uniform(0.5, 1.5),
                    f * np.random.uniform(1.0 / 5.0, 1.0),
                    np.random.uniform(50, 25),
                    np.random.uniform(12.5, 6.25),
                    np.random.uniform(3.75, 1.0 / 10.0),
                ][np.random.choice([0, 1, 2, 3, 4], p=[0.3, 0.25, 0.2, 0.15, 0.1])]
            ) / sr
            assert sharpness_blip > 0
            signal_copy += (
                wave
                * bell_curve(
                    indices_signal_samples,
                    loc_blip,
                    sharpness_blip,
                    fat_tails=np.random.uniform(0, 1) < 0.125,
                )
                * (
                    utils.map_range(
                        np.log2(sharpness_blip * sr),
                        (np.log2(100), np.log2(1000)),
                        (2, 3),
                    )
                )  # 2 because sqrt(2) for phase cancellation and sqrt(2) for taper compensation
            )

    if n_unsmooths > 0:
        fr_dB = np.zeros((4095,), dtype=np.float64)
        bindices = np.arange(4095)
        for _ in range(n_unsmooths):
            multiplier_dB = np.random.uniform(-20, 10)
            f_Hz = np.random.uniform(2500, f_Hz_nyquist)
            f_bindex = f_Hz / f_Hz_nyquist * 4096 - 1
            breadth_bins = np.random.uniform(
                1 / 2, utils.map_range(f_Hz, (5000, 15000), (3.75, 7.5))  # radii
            )
            fr_dB += multiplier_dB * step_bandpass(bindices - f_bindex, breadth_bins)
        ir = utils.make_ir(
            librosa.db_to_amplitude(fr_dB),
            1 / 10,
            sr,
        )
        signal_copy = utils.center_kernel_convolve_segment(0, l, signal_copy, ir)

    if (not (aug_harsh_signalbank is None)) and ENUM_ARTIFACTS[
        "sampled-harshness"
    ] in artifacts_picked:  # np.random.uniform(0, 1) <= 1 # 1/25
        fraction = 5 * (np.random.uniform(0, 1) ** 2)
        outshift = l / max(1, l_s * fraction)
        # signal_copy += fetch_chunk(aug_harsh_signalbank, l) * 0.01
        # print(signal_copy.dtype)
        # """
        # print(l - np.sum(np.isfinite(fetch_chunk(aug_harsh_signalbank, l))))
        signal_copy += (
            fetch_chunk(aug_harsh_signalbank, l)
            * librosa.db_to_amplitude(np.random.uniform(-37.5, -18.75))
            * np.random.choice(POLARITIES)
            * np.sqrt(
                2
            )  # not 2 because these bell curves last way longer than the blips’ bell curves
            * bell_curve(
                indices_signal_samples,
                np.random.uniform(-outshift, l + outshift),
                fraction / sr,  # 0 Hz to 10 Hz
                fat_tails=True,
            )
        )
        # """

    if ENUM_ARTIFACTS["comb"] in artifacts_picked:
        comb_filter_lowest_null_f_Hz = 3200 * 2 ** (np.random.normal(0, 3 / 4))
        comb_filter_fade_low_bound_f_Hz = min(
            15000, 10000 * 2 ** np.random.normal(0, 1 / 2)
        )
        comb_filter_fade_distance_Hz = f_Hz_nyquist - comb_filter_fade_low_bound_f_Hz
        comb_filter_mix_amount = np.random.uniform(
            1 - librosa.db_to_amplitude(-12.5), 1
        )
        comb_fr = np.sqrt(
            (
                1
                + np.cos(
                    (2 * np.pi) / (2 * comb_filter_lowest_null_f_Hz) * fr_x_Hz_per_bin
                )
            )
            / 2
        )
        comb_fr = 1 - comb_fr
        comb_fr *= comb_filter_mix_amount
        comb_fr *= (
            1
            + np.tanh(
                (
                    (fr_x_Hz_per_bin - comb_filter_fade_low_bound_f_Hz)
                    / (comb_filter_fade_distance_Hz / np.random.uniform(3, 4))
                )
                - 1
            )
        ) / 2
        comb_fr = 1 - comb_fr
        do_shifting_comb = np.random.uniform(0, 1) < 1 / 5
        if not do_shifting_comb:
            comb_ir = utils.make_ir(
                comb_fr,
                1 / 5,
                sr,
            )
            signal_copy = utils.center_kernel_convolve_segment(
                0, l, signal_copy, comb_ir
            )
        else:
            LEN_GRAIN_SAM = 2400
            LEN_HOP_SAM = 1200
            manager_grains = overlap_add_2.BlockProcessManager(
                signal_copy, LEN_GRAIN_SAM, 2
            )
            interpolator_comb_fr_amp = PchipInterpolator(
                *utils.const_extrap(fr_x_Hz_per_bin, comb_fr), extrapolate=True
            )
            blocks_iterable = sorted(
                manager_grains.blocks_iterable(), key=lambda g: g.bounds[0]
            )
            fr_pitch_shift_per_block = np.random.uniform(0.0, 1.0, len(blocks_iterable))
            fr_pitch_shift_per_block = utils.fast_blurfilter(
                fr_pitch_shift_per_block,
                2.0 ** np.random.uniform(-3.5, 0.0) * sr / LEN_HOP_SAM,
            )
            fr_pitch_shift_per_block = minmax_scale(
                fr_pitch_shift_per_block, (0.0, 1.0)
            )
            fr_pitch_shift_per_block = (
                utils.inverse_smoothstep(fr_pitch_shift_per_block) - 1.0
            )
            fr_pitch_shift_per_block = (2.0 ** (1.0 / 12.0)) ** fr_pitch_shift_per_block
            for shift_pitch_this_block, subsig in zip(
                fr_pitch_shift_per_block, blocks_iterable
            ):
                # print(shift_pitch_this_block)
                ir = utils.make_ir(
                    interpolator_comb_fr_amp(shift_pitch_this_block * fr_x_Hz_per_bin),
                    1 / 10,
                    sr,
                )
                subsig.set(subsig.convolve(ir))
            manager_grains.write_to_base()
            signal_copy = manager_grains.base

    return signal_copy


def trimmed_spectrogram(signal, l_x_hops, l_win_fft, height, is_for_training=True):
    # trims to get rid of boundaries’ effects
    gram = signal_to_spectrogram(
        signal, (l_x_hops, height), l_win_fft, is_for_training=is_for_training
    )

    # gram[:padding] = np.tile(gram[padding], (padding, 1))
    # gram[-padding:] = np.tile(gram[-padding - 1], (padding, 1))
    # print(l_x_hops, gram.shape[0])

    if is_for_training:
        assert gram.shape[0] == l_x_hops

    gram = resize(
        gram, (gram.shape[0], 1 + height)
    )  # librosa’s stfts’ returns’ shapes = 1 + l_win_fft / 2

    gram = (
        dB_hearing_scale_softplus(utils.amplitude_to_decibels(gram))
        if is_for_training
        else gram
    )

    return gram[
        :, 1:
    ]  # signal_to_spectrogram(signal, l_x_hops + 2 * padding)[padding:-padding]


def cropped_spectrogram(signal, l_x_hops_b4, l_x_hops_aftr, l_win_fft, height):
    DIM_TIME_CROP_PER_SIDE = (l_x_hops_b4 - l_x_hops_aftr) / 2
    assert (
        utils.is_an_int(DIM_TIME_CROP_PER_SIDE)
        # and DIM_TIME_CROP_PER_SIDE >= n_hops_offset
    )
    DIM_TIME_CROP_PER_SIDE = int(DIM_TIME_CROP_PER_SIDE)
    return trimmed_spectrogram(signal, l_x_hops_b4, l_win_fft, height)[
        DIM_TIME_CROP_PER_SIDE:-DIM_TIME_CROP_PER_SIDE
    ]


#
__octaves_below_nyquist = np.tile(
    np.log2(
        (np.arange(1, 1 + DIMS_CROPPED[-1][1]) + (1.0 / 2.0)) / DIMS_CROPPED[-1][1]
    ),
    (DIMS_CROPPED[-1][0], 1),
)
# print(__octaves_below_nyquist.shape)
SPECT_CASCADE_MASKS = (
    np.clip(__octaves_below_nyquist, a_min=-2, a_max=-1) + 2,
    (1 - (np.clip(__octaves_below_nyquist, a_min=-2, a_max=-1) + 2))
    * (np.clip(__octaves_below_nyquist, a_min=-4, a_max=-3) + 4),
    1 - (np.clip(__octaves_below_nyquist, a_min=-4, a_max=-3) + 4),
)

#


def get_train_pair(
    signal,
    sr,
    aug_noise_signal=None,
    aug_filtered_noise_signal=None,
    aug_harsh_signal=None,
):
    chunk_good = fetch_chunk(
        signal,
        int(DUR_TIME_OVERSCAN_S * sr),
        augmentation_src_sig=aug_noise_signal,
        augmentation_src_sig_prefiltered=aug_filtered_noise_signal,
    )
    chunk_bad = chunk_good
    if np.random.uniform(0, 1) < PROPORTION_IN_DATASET__ARITFACTS:
        chunk_bad = alter_chunk(chunk_good, sr, aug_harsh_signalbank=aug_harsh_signal)

    chunk_good = resonot_adapter.apply_to_monoch(chunk_good, sr)

    spectrogram_in_1024 = cropped_spectrogram(
        chunk_bad, DIMS_OVERSCAN[-1][0], DIMS_CROPPED[-1][0], 1024, 1024
    )
    spectrogram_out_1024 = cropped_spectrogram(
        chunk_good, DIMS_OVERSCAN[-1][0], DIMS_CROPPED[-1][0], 1024, 1024
    )
    spectrogram_in_2048 = cropped_spectrogram(
        chunk_bad, DIMS_OVERSCAN[-1][0], DIMS_CROPPED[-1][0], 2048, 1024
    )
    spectrogram_out_2048 = cropped_spectrogram(
        chunk_good, DIMS_OVERSCAN[-1][0], DIMS_CROPPED[-1][0], 2048, 1024
    )
    spectrogram_in_4096 = cropped_spectrogram(
        chunk_bad, DIMS_OVERSCAN[-1][0], DIMS_CROPPED[-1][0], 4096, 1024
    )
    spectrogram_out_4096 = cropped_spectrogram(
        chunk_good, DIMS_OVERSCAN[-1][0], DIMS_CROPPED[-1][0], 4096, 1024
    )

    n_hops, n_bins = spectrogram_in_1024.shape
    Hz_per_bin = (np.arange(1, 1 + n_bins) + (1.0 / 2.0)) / n_bins * sr / 2.0

    def residual(x, y):
        _residual_dB = gaussian_filter(
            y - x, sigma=1 / 2
        ) - utils.amplitude_to_decibels(
            np.sqrt(
                gaussian_filter(
                    utils.decibels_to_amplitude(x - y) ** 2,
                    sigma=1 / 2,
                )
            )
        )
        _residual_dB /= 2
        return _residual_dB

    residual_1024 = residual(spectrogram_in_1024, spectrogram_out_1024)
    residual_2048 = residual(spectrogram_in_2048, spectrogram_out_2048)
    residual_4096 = residual(spectrogram_in_4096, spectrogram_out_4096)

    """
    mask_4096 = utils.map_range_vector(
        Hz_per_bin, (125, 250), (0, 1)
    ) * utils.map_range_vector(Hz_per_bin, (500, 2500), (1, 0))
    mask_2048 = utils.map_range_vector(
        Hz_per_bin, (500, 2500), (0, 1)
    ) * utils.map_range_vector(Hz_per_bin, (5000, 10000), (1, 0))
    mask_1024 = utils.map_range_vector(
        Hz_per_bin, (5000, 10000), (0, 1)
    ) * utils.map_range_vector(Hz_per_bin, (22050, 24000), (1, 0))
    """

    def spect_cascade(Y1024, Y2048, Y4096):
        return (
            Y1024 * SPECT_CASCADE_MASKS[0]
            + Y2048 * SPECT_CASCADE_MASKS[1]
            + Y4096 * SPECT_CASCADE_MASKS[2]
        )

    composite_residual_dB = spect_cascade(residual_1024, residual_2048, residual_4096)
    composite_residual_dB_boosts_halved = residual_1024 * np.where(
        residual_1024 > 0, 1 / 2, 1
    )
    composite_residual_dB = utils.map_range_vector(
        np.tile(Hz_per_bin, (n_hops, 1)),
        (22050.0, 24000.0),
        (composite_residual_dB, composite_residual_dB_boosts_halved),
    ) * utils.map_range_vector(
        np.tile(utils.hertz_to_C_octaves(Hz_per_bin), (n_hops, 1)),
        (utils.hertz_to_C_octaves(50), utils.hertz_to_C_octaves(125)),
        (0.0, 1.0),
    )
    """(
        residual_1024 * np.tile(mask_1024, (n_hops, 1))
        + residual_2048 * np.tile(mask_2048, (n_hops, 1))
        + residual_4096 * np.tile(mask_4096, (n_hops, 1))
    )"""
    DELTA_MAGNITUDE_MIN_DB = 0.125
    composite_residual_dB -= np.clip(
        composite_residual_dB, -DELTA_MAGNITUDE_MIN_DB, DELTA_MAGNITUDE_MIN_DB
    )  # changes smaller than DELTA_MAGNITUDE_MIN_DB dB are counted as nil
    composite_residual_dB = np.clip(composite_residual_dB, -30, 15)
    composite_residual_dB = gaussian_filter(composite_residual_dB, sigma=1 / 2)

    return [
        cropped_spectrogram(
            chunk_bad, DIMS_OVERSCAN[0][0], DIMS_CROPPED[0][0], 1024, DIMS_CROPPED[0][1]
        ),
        cropped_spectrogram(
            chunk_bad, DIMS_OVERSCAN[1][0], DIMS_CROPPED[1][0], 2048, DIMS_CROPPED[1][1]
        ),
        cropped_spectrogram(
            chunk_bad, DIMS_OVERSCAN[2][0], DIMS_CROPPED[2][0], 4096, DIMS_CROPPED[2][1]
        ),
    ], composite_residual_dB


if __name__ == "__main__":

    lut = resonot_adapter.lut()

    import os, time
    import soundfile as sf
    from tqdm import tqdm

    MOUNT_POINT = "/Volumes/[redacted]////"  # "./"

    augment_ambience_src, augment_ambience_sr = sf.read(
        MOUNT_POINT + "train_data_sources/augmentations/background_noises/[redacted]"
    )
    augment_ambience_src_prefiltered = preprocess_source(
        augment_ambience_src, augment_ambience_sr
    )

    augment_harsh_src, augment_harsh_sr = sf.read(
        MOUNT_POINT + "train_data_sources/augmentations/harsh_sounds/[redacted]"
    )
    augment_harsh_src = preprocess_source(augment_harsh_src, augment_harsh_sr)

    print(augment_ambience_src.dtype)
    print(augment_ambience_src_prefiltered.dtype)
    print(augment_harsh_src.dtype)

    assert SR == augment_ambience_sr and SR == augment_harsh_sr

    DATASET_SIZE_N_PAIRS = (
        6500  # ~<s>32<s>K to cover the same amount of time as the source material
    )

    N_HUGE_AUDIO_PARTS = 13

    N_PAIRS_PER_PART = DATASET_SIZE_N_PAIRS // N_HUGE_AUDIO_PARTS

    dataset_real_size = N_HUGE_AUDIO_PARTS * N_PAIRS_PER_PART
    folder_out_no_ts = f"train_{dataset_real_size}_splits_{N_HUGE_AUDIO_PARTS}"
    folder_out = (
        "train_data_packs_1s_multires/" + folder_out_no_ts + f" ({time.time()})"
    )

    if not os.path.exists(MOUNT_POINT + folder_out):
        os.makedirs(MOUNT_POINT + folder_out)
    if not os.path.exists(MOUNT_POINT + folder_out + "/x1"):
        os.makedirs(MOUNT_POINT + folder_out + "/x1")
    if not os.path.exists(MOUNT_POINT + folder_out + "/x2"):
        os.makedirs(MOUNT_POINT + folder_out + "/x2")
    if not os.path.exists(MOUNT_POINT + folder_out + "/x3"):
        os.makedirs(MOUNT_POINT + folder_out + "/x3")
    if not os.path.exists(MOUNT_POINT + folder_out + "/y"):
        os.makedirs(MOUNT_POINT + folder_out + "/y")

    src = None

    for i_part in range(N_HUGE_AUDIO_PARTS):
        src, sr = sf.read(
            MOUNT_POINT + f"train_data_sources/base/giga_parts/{i_part}.wav"
        )
        assert SR == sr
        src = preprocess_source(src, sr)

        X1s = []
        X2s = []
        X3s = []
        Ys = []

        for j in tqdm(range(N_PAIRS_PER_PART)):
            (x1, x2, x3), y = get_train_pair(
                src,
                sr,
                augment_ambience_src,
                augment_ambience_src_prefiltered,
                augment_harsh_src,
            )
            X1s.append(x1)
            X2s.append(x2)
            X3s.append(x3)
            Ys.append(y)

        X1s = np.array(X1s, dtype=np.float16)
        X2s = np.array(X2s, dtype=np.float16)
        X3s = np.array(X3s, dtype=np.float16)
        Ys = np.array(Ys, dtype=np.float16)
        assert np.all(X1s.shape == np.array([N_PAIRS_PER_PART, *(DIMS_CROPPED[0])]))
        assert np.all(X2s.shape == np.array([N_PAIRS_PER_PART, *(DIMS_CROPPED[1])]))
        assert np.all(X3s.shape == np.array([N_PAIRS_PER_PART, *(DIMS_CROPPED[2])]))
        assert np.all(Ys.shape == np.array([N_PAIRS_PER_PART, *(DIMS_CROPPED[-1])]))

        pair_save_timestamp = time.time()

        np.save(
            MOUNT_POINT
            + f"{folder_out}/x1/{folder_out_no_ts.replace('train_','train_x1_')}---{i_part} ({pair_save_timestamp}).npy",
            X1s,
        )
        np.save(
            MOUNT_POINT
            + f"{folder_out}/x2/{folder_out_no_ts.replace('train_','train_x2_')}---{i_part} ({pair_save_timestamp}).npy",
            X2s,
        )
        np.save(
            MOUNT_POINT
            + f"{folder_out}/x3/{folder_out_no_ts.replace('train_','train_x3_')}---{i_part} ({pair_save_timestamp}).npy",
            X3s,
        )
        np.save(
            MOUNT_POINT
            + f"{folder_out}/y/{folder_out_no_ts.replace('train_','train_y_')}---{i_part} ({pair_save_timestamp}).npy",
            Ys,
        )

    resonot_adapter.shutdown()
